import gymnasium as gym
import random
import numpy as np
import argparse
import torch
import yaml
from pathlib import Path
import os
import wandb
import datetime 
import random
import time 
from rpc import RPCAgent 
from rrpc import RRPCAgent

def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='RPC')
    parser.add_argument('--env_name', type=str, default=config['env_name'])
    parser.add_argument('--device', type=str, default=config['device'])
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--num_train_steps', type=int, default=config['num_train_steps'])
    parser.add_argument('--learning_starts', type=int, default=config['learning_starts'])
    parser.add_argument('--gamma', type=float, default=config['gamma'])
    parser.add_argument('--tau', type=float, default=config['tau'])

    parser.add_argument('--env_buffer_size', type=int, default=config['env_buffer_size'])

    parser.add_argument('--target_update_interval', type=int, default=config['target_update_interval'])
    parser.add_argument('--log_interval', type=int, default=config['log_interval'])
    parser.add_argument('--save_snapshot_interval', type=int, default=config['save_snapshot_interval'])
    parser.add_argument('--eval_episode_interval', type=int, default=config['eval_episode_interval'])
    parser.add_argument('--num_eval_episodes', type=int, default=config['num_eval_episodes'])
    
    

    parser.add_argument('--latent_dims', type=int, default=config['latent_dims'])
    parser.add_argument('--model_hidden_dims', type=int, default=config['model_hidden_dims'])
    parser.add_argument('--model_num_layers', type=int, default=config['model_num_layers'])
    parser.add_argument('--kl_constraint', type=float, default=config['kl_constraint'])
    parser.add_argument('--lambda_init', type=float, default=config['lambda_init'])
    parser.add_argument('--alpha_autotune', type=str, default=config['alpha_autotune'])
    parser.add_argument('--alpha_init', type=float, default=config['alpha'])
    parser.add_argument('--noise_factor', type=float, default=config['noise_factor'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--seq_len', type=int, default=config['seq_len'])
    parser.add_argument('--lr', type=float, default=config['lr'])
    
    parser.add_argument('--record_eval', action='store_true', help='save & log eval videos', default=True)
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--video_fps', type=int, default=30)
    
    
    parser.add_argument('--use_rpc', type=int, default=1)
    

    
    args = parser.parse_args()
    return args

def make_agent(env, device, args):
    num_states = np.prod(env.observation_space.shape)
    num_actions = np.prod(env.action_space.shape)
    
    if args.agent == 'RPC':
        agent = RPCAgent(env, device, num_states, num_actions, args.gamma, args.tau, 
                            args.env_buffer_size, args.target_update_interval,
                            args.log_interval, args.latent_dims, args.model_hidden_dims, 
                            args.model_num_layers, args.kl_constraint, args.lambda_init,
                            args.alpha_init, args.alpha_autotune, 
                            args.batch_size, args.lr, use_rpc=args.use_rpc == 1)

    if args.agent == 'RRPC':
        agent = RRPCAgent(env, device, num_states, num_actions, args.gamma, args.tau, 
                            args.env_buffer_size, args.target_update_interval,
                            args.log_interval, args.latent_dims, args.model_hidden_dims, 
                            args.kl_constraint, args.lambda_init,
                            args.alpha_init, args.alpha_autotune, args.seq_len,
                            args.batch_size, args.lr)
    
    return agent

class MujocoWorkspace:
    def __init__(self, args, run_tag):
        self.args = args
        self.run_tag = run_tag
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device=='cuda' else "cpu")
        print(self.device)
        self.work_dir = Path.cwd()
        self.setup()
        self.set_seeds_everywhere()
        
        print(self.args)
        
        self.agent = make_agent(self.train_env, self.device, self.args)
        self._global_step = 0
        self._global_episode = 0
        self._best_eval_returns = -np.inf
        


    def setup(self):
        # self.train_env = gym.make(self.args.env_name)
        # self.eval_env = gym.make(self.args.env_name)        
        # self.robust_env = gym.make(self.args.env_name)
        # self.checkpoint_path = os.path.join(self.work_dir,'checkpoints/' + self.args.agent +'_' + self.args.env_name + '/' + str(datetime.datetime.now())) 
        # os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.train_env  = gym.make(self.args.env_name)  # no rendering needed here
        self.eval_env   = gym.make(self.args.env_name, render_mode="rgb_array")
        self.robust_env = gym.make(self.args.env_name, render_mode="rgb_array")
        self.checkpoint_path = os.path.join(
                self.work_dir, 'checkpoints',
                f"{self.args.agent}_{self.args.env_name}",
                self.run_tag + '_' + wandb.run.id
                )
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # videos folder
        Path(self.args.video_dir).mkdir(parents=True, exist_ok=True)
        
    def _kl_target_bits(self):
        return float(self.args.kl_constraint)
        
    def _dump_args_if_missing(self, directory):
        from pathlib import Path
        import yaml
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "args.yaml"
        if not p.exists():
            with p.open("w") as f:
                yaml.safe_dump(vars(self.args), f, sort_keys=False)



            
    def _write_run_meta(self, dst):
        import yaml, wandb
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "args.yaml").write_text(yaml.safe_dump(vars(self.args), sort_keys=True))
        (dst / "wandb.txt").write_text(f"id={getattr(wandb.run,'id',None)}\nname={getattr(wandb.run,'name',None)}\n")

    def _save_and_log_video(self, frames, tag):
        if not frames:
            return
        import imageio.v2 as imageio
        import numpy as np
        from pathlib import Path
        from datetime import datetime
        import wandb

        arr = np.stack(frames, axis=0).astype(np.uint8)
        
        
        base = Path(self.args.video_dir) / self.args.env_name / f"{self.run_tag}_{wandb.run.id}"
        base.mkdir(parents=True, exist_ok=True)

        self._write_run_meta(base)
        
        kl_bits = self._kl_target_bits()
        kl_tag  = f"kl{kl_bits}" if kl_bits is not None else "klNA"

        out_path = base / f"{tag}_step{self._global_step}_{kl_tag}.mp4"
        imageio.mimsave(out_path, arr, fps=int(self.args.video_fps), codec="libx264", format="ffmpeg")

        payload = {
            # f"{tag}/video": wandb.Video(str(out_path), fps=int(self.args.video_fps), format="mp4"),
            f"{tag}/video_path": str(out_path),
        }
        


        
        
        if kl_bits is not None:
            payload[f"{tag}/kl_target_bits"] = kl_bits
        wandb.log(payload, step=self._global_step)

        print(f"[{self.args.env_name}] saved {out_path}")






    def set_seeds_everywhere(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        self.train_env.reset(seed=self.args.seed)
        self.train_env.action_space.seed(self.args.seed)
        self.train_env.observation_space.seed(self.args.seed)
        self.eval_env.reset(seed = self.args.seed)
        self.eval_env.action_space.seed(self.args.seed)
        self.eval_env.observation_space.seed(self.args.seed)
        self.robust_env.reset(seed=self.args.seed)
        self.robust_env.action_space.seed(self.args.seed)
        self.robust_env.observation_space.seed(self.args.seed)

    def train(self):
        (state, info), done, episode_return, episode_length = self.train_env.reset(), False, 0., 0
        for _ in range(self._global_step, self.args.num_train_steps+1):  
            if self._global_step <= self.args.learning_starts:
                action = self.train_env.action_space.sample()
            else:
                action, _ = self.agent.get_action(state)

            next_state, reward, terminated, truncated, info = self.train_env.step(action)
            # print('nextstate', type(next_state))
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

            if done and episode_length == self.train_env._max_episode_steps:
                true_done = False 
            else:
                true_done = done

            self.agent.env_buffer.push((state, action, reward, next_state, true_done))

            duration_step = None
            
            if len(self.agent.env_buffer) > self.args.batch_size and self._global_step > self.args.learning_starts:
                start = time.time()
                self.agent.update(self._global_step)
                duration_step = time.time() - start
            
            if (self._global_step+1)%self.args.eval_episode_interval==0:
                self.eval()
                self.eval_robustness()

            if self._global_step%self.args.save_snapshot_interval==0:
                self.save_snapshot()

            self._global_step += 1
            if done:
                self._global_episode += 1
                print("Episode: {}, total numsteps: {}, return: {}".format(self._global_episode, self._global_step, round(episode_return, 2)))
                episode_metrics = {}
                if self._global_step > self.args.learning_starts:
                    if duration_step is not None:  # <-- only log if we actually updated
                        episode_metrics["duration_step"] = duration_step
                    
                    
                episode_metrics['episodic_length'] = episode_length
                episode_metrics['episodic_return'] = episode_return
                episode_metrics['env_buffer_length'] = len(self.agent.env_buffer)

                wandb.log(episode_metrics, step=self._global_step)
                (state, info), done, episode_return, episode_length = self.train_env.reset(), False, 0., 0
            else:
                state = next_state
                
        self.train_env.close()
 
    def eval(self):
        steps, returns = 0, 0
        record_first = self.args.record_eval
        frames = []  # only for the first episode

        for e in range(self.args.num_eval_episodes):
            done = False
            state, info = self.eval_env.reset()
            if record_first and e == 0:
                frame = self.eval_env.render()   # (H,W,C) uint8
                if frame is not None:
                    frames.append(frame)

            while not done:
                with torch.no_grad():
                    action, _ = self.agent.get_action(state, True)

                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)

                returns += float(reward)
                steps += 1
                state = next_state

                if record_first and e == 0:
                    frame = self.eval_env.render()
                    if frame is not None:
                        frames.append(frame)

        avg_return = returns / self.args.num_eval_episodes
        if avg_return >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = avg_return

        wandb.log({
            'eval/episodic_return': avg_return,
            'eval/episodic_length': steps / self.args.num_eval_episodes
        }, step=self._global_step)

        if record_first:
            self._save_and_log_video(frames, tag="eval")


    def eval_robustness(self):
        steps, returns = 0, 0

        robust_evals = 2
        for _ in range(robust_evals):
            done = False 
            state, info = self.robust_env.reset()
            while not done:
                r = random.uniform(0.0, 1.0)
                if r>0.5:
                    state[2] += self.args.noise_factor*(np.random.rand()-0.5) 
                with torch.no_grad():
                    action, _ = self.agent.get_action(state, True)
                    
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
                
                returns += reward
                steps += 1
                state = next_state

        robust_metrics = {}
        robust_metrics['robust_episodic_return'] = returns / robust_evals
        robust_metrics['robust_episodic_length'] = steps / robust_evals
        wandb.log(robust_metrics, step = self._global_step)
        
    def eval_openloop(self):
        steps, returns = 0, 0
        
        eval_open = 3
        p = 0.1

        for _ in range(eval_open):
            done = False 
            state, info = self.robust_env.reset()
            last_z_dist = None
            while not done:
                r = random.uniform(0.0, 1.0)
                
                if r < p and last_z != None:
                    
                    action, z_pred = self.agent.open_loop_action(last_z)
                    last_z = z_pred
                    
                    
                with torch.no_grad():
                    action, _ = self.agent.get_action(state, True)
                    last_z = self.agent.get_state_latent(state)
                    
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
                
                returns += reward
                steps += 1
                state = next_state

        robust_metrics = {}
        robust_metrics['openloop_episodic_return'] = returns / eval_open
        robust_metrics['openloop_episodic_length'] = steps / eval_open
        wandb.log(robust_metrics, step = self._global_step)

    def save_snapshot(self, best=False):
        fname = 'best.pt' if best else f'{self._global_step}.pt'
        path = Path(self.checkpoint_path) / fname
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "args": vars(self.args),
            "_global_step": self._global_step,
            "_global_episode": self._global_episode,
            "meta": {"env_name": self.args.env_name},

            "encoder": self.agent.encoder.state_dict(),
            "actor":   self.agent.actor.state_dict(),
            "model":   self.agent.model.state_dict(),
        }

        torch.save(payload, path)
        print(f"[save_snapshot] saved {path}")


    def load_snapshot(self, path, iter='best', map_location=None, strict=True):
        fname = 'best.pt' if iter == 'best' else f'{int(iter)}.pt'
        path = Path(path) / fname
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"No checkpoint at {path}")
            print(f"No checkpoint at {path}")
            return False

        device = self.device
        payload = torch.load(path, map_location=(map_location or device))

        # restore weights
        self.agent.encoder.load_state_dict(payload["encoder"])
        self.agent.model.load_state_dict(payload["model"])
        self.agent.actor.load_state_dict(payload["actor"])
        self.agent.encoder.to(device).eval()
        self.agent.model.to(device).eval()
        self.agent.actor.to(device).eval()

        # restore counters and args
        self._global_step = payload.get("_global_step", 0)
        self._global_episode = payload.get("_global_episode", 0)
        self.args = argparse.Namespace(**payload.get("args", {}))
        
        # re initialize vars
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.device=='cuda' else "cpu")
        self.work_dir = Path.cwd()
        self.setup()
        self.set_seeds_everywhere()

        self.agent = make_agent(self.train_env, self.device, self.args)
        self._best_eval_returns = -np.inf

        print(f"[load_snapshot] loaded {path} @ step {self._global_step}")
        return True


        
def main():

    with open("mujoco.yaml", 'r') as stream:
        mujoco_config = yaml.safe_load(stream)
    args = parse_args(mujoco_config['rpc_params'])
    
    if not args.use_rpc:
        args.kl_constraint = 0
    
    # generate run tag
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    kl_tag = f"kl_{args.kl_constraint:.3g}" if args.kl_constraint is not None else "klNA"
    run_tag = f"{ts}_{args.agent}_{args.env_name}_seed_{args.seed}_{kl_tag}"
    
    print(run_tag)



    with wandb.init(
        project="rpc",
        entity="arjaras-university-of-pennsylvania",
        group=args.env_name,
        resume="allow",
        config=vars(args),
    ):
        print(f"[INFO] Using run: {wandb.run.name} ({wandb.run.id})")

        workspace = MujocoWorkspace(args, run_tag=run_tag)
        # workspace.load_snapshot(path='./checkpoints/RPC_Walker2d-v5/RPC_1_kl1_ix8in5b7', iter=9500)
        workspace.train()



if __name__ == '__main__':
    main()