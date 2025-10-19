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
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device=='cuda' else "cpu")
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
                wandb.run.id
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



            
    def _save_and_log_video(self, frames, tag):
        if not frames:
            return

        import imageio.v2 as imageio
        import numpy as np
        import datetime
        from pathlib import Path
        import wandb

        arr = np.stack(frames, axis=0).astype(np.uint8)

        # KL bits tag for naming
        kl_bits = self._kl_target_bits()
        kl_tag  = f"kl{kl_bits}" if kl_bits is not None else "klNA"

        # unique run id
        run_id = wandb.run.id if wandb.run is not None \
                else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 🔑 New: env-level subdir → then run subdir
        env_dir = Path(self.args.video_dir) / self.args.env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        video_subdir = env_dir / f"{self.args.agent}_{self.args.seed}_{kl_tag}_{run_id}"
        video_subdir.mkdir(parents=True, exist_ok=True)

        # NEW: save args.yaml in the video folder if missing
        self._dump_args_if_missing(video_subdir)

        out_path = video_subdir / f"{tag}_step{self._global_step}_{kl_tag}.mp4"

        imageio.mimsave(
            out_path,
            arr,
            fps=int(self.args.video_fps),
            codec="libx264",
            format="ffmpeg"
        )

        payload = {
            f"{tag}/video": wandb.Video(str(out_path), fps=int(self.args.video_fps), format="mp4"),
            f"{tag}/video_path": str(out_path),
        }
        if kl_bits is not None:
            payload[f"{tag}/kl_target_bits"] = kl_bits

        wandb.log(payload, step=self._global_step)

        # 🔑 Optional: print env dirs sorted by modified time
        # (helps you see most recent runs first in Explorer / logs)
        dirs_sorted = sorted(env_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"[{self.args.env_name}] runs sorted by time:")
        for d in dirs_sorted[:5]:  # only show last 5 for readability
            print("  ", d.name)





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
                    action = self.agent.get_action(state, True)

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
                    action = self.agent.get_action(state, True)
                    
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

        for _ in range(1):
            done = False 
            state, info = self.robust_env.reset()
            while not done:
                r = random.uniform(0.0, 1.0)
                if r>0.5:
                    state[2] += self.args.noise_factor*(np.random.rand()-0.5) 
                with torch.no_grad():
                    action = self.agent.get_action(state, True)
                    
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
                
                returns += reward
                steps += 1
                state = next_state

        robust_metrics = {}
        robust_metrics['robust_episodic_return'] = returns
        robust_metrics['robust_episodic_length'] = steps
        wandb.log(robust_metrics, step = self._global_step)

    def save_snapshot(self, best=False):
        keys_to_save = ['agent', '_global_step', '_global_episode', 'args']
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._global_step)+'.pt')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, iter, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(iter)+'.pt')
            
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
            
    def load_best(self, path):
        snapshot = Path(path, 'best.pt')
        with snapshot.open('rb') as f:
            payload = torch.load(f, weights_only=False)
        for k, v in payload.items():
            self.__dict__[k] = v

    def salvage_weights(self):
        from pathlib import Path



        st = 'robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_11-36-23.998758 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_11-50-15.741457 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_11-53-56.534500 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_17-07-53.558064 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_17-09-47.965958 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_17-20-19.030783 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_22-30-46.464062 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_22-30-58.504174 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-01_22-44-17.106398 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_03-46-17.646129 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_03-51-56.396265 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_04-04-00.644838 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_09-04-37.596086 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_09-10-15.413442 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_09-19-08.051633 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_14-23-04.267465 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_14-29-54.712829 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-02_14-39-58.759492 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-15_15-15-47.359893 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-15_15-17-38.593985 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-15_15-18-32.309093 robust-predictable-control/checkpoints/RPC_Walker2d-v5/2025-10-15_16-03-05.735273'
        paths = st.split(' ')
        print(len(paths))

        

        for exp_path in paths:
            p = Path('../' + exp_path)

            snapshot = Path(p, 'best.pt')
            with snapshot.open('rb') as f:
                payload = torch.load(f, weights_only=False)
            for k, v in payload.items():
                self.__dict__[k] = v
                
            print(self.agent.kl_constraint)
        
def main():

    with open("mujoco.yaml", 'r') as stream:
        mujoco_config = yaml.safe_load(stream)
    args = parse_args(mujoco_config['rpc_params'])
    
    import os, wandb


    if "WANDB_RUN_ID" in os.environ:
        run_id = os.environ["WANDB_RUN_ID"]
    else:
        # e.g. "RPC_seed0_kl0.05-bright-surf-7"
        human_name = wandb.util.generate_id()
        run_id = f"test_eval_{human_name}"
        os.environ["WANDB_RUN_ID"] = run_id
        
    with wandb.init(
        project="rpc",
        entity="arjaras-university-of-pennsylvania",
        group=args.env_name,
        id=run_id,
        resume="allow",
        config=args.__dict__,
    ):
        wandb.run.name = run_id
        wandb.config.update({"run_id": run_id}, allow_val_change=True)

        print(f"[INFO] Using run_id / name: {run_id}")

        workspace = MujocoWorkspace(args)
        workspace.salvage_weights()


if __name__ == '__main__':
    main()