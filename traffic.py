import gymnasium as gym
import highway_env
import random
import numpy as np
import argparse
import torch
import yaml
from pathlib import Path
import os
import wandb
import datetime
import time

from rpc import RPCAgent
from rrpc import RRPCAgent


def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='RPC')
    parser.add_argument('--env_name', type=str, default=config['env_name'])  # e.g. "highway-v0"
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

    return parser.parse_args()


def make_agent(env, device, args):
    num_states = np.prod(env.observation_space.shape)
    num_actions = env.action_space.n  # always discrete for highway

    if args.agent == 'RPC':
        return RPCAgent(env, device, num_states, num_actions, args.gamma, args.tau,
                        args.env_buffer_size, args.target_update_interval,
                        args.log_interval, args.latent_dims, args.model_hidden_dims,
                        args.model_num_layers, args.kl_constraint, args.lambda_init,
                        args.alpha_init, args.alpha_autotune,
                        args.batch_size, args.lr)

    elif args.agent == 'RRPC':
        return RRPCAgent(env, device, num_states, num_actions, args.gamma, args.tau,
                         args.env_buffer_size, args.target_update_interval,
                         args.log_interval, args.latent_dims, args.model_hidden_dims,
                         args.kl_constraint, args.lambda_init,
                         args.alpha_init, args.alpha_autotune, args.seq_len,
                         args.batch_size, args.lr)
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")


class HighwayWorkspace:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
        self.work_dir = Path.cwd()
        self.setup()
        self.set_seeds()
        self.agent = make_agent(self.train_env, self.device, self.args)
        self._global_step = 0
        self._global_episode = 0
        self._best_eval_returns = -np.inf

    def setup(self):
        self.train_env = gym.make(self.args.env_name)
        self.eval_env = gym.make(self.args.env_name, render_mode="rgb_array")
        self.robust_env = gym.make(self.args.env_name, render_mode="rgb_array")

        self.checkpoint_path = Path(self.work_dir, "checkpoints",
                                    f"{self.args.agent}_{self.args.env_name}",
                                    str(datetime.datetime.now()))
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        Path(self.args.video_dir).mkdir(parents=True, exist_ok=True)

    def set_seeds(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        for env in [self.train_env, self.eval_env, self.robust_env]:
            env.reset(seed=self.args.seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(self.args.seed)

    def train(self):
        (state, info), done, ep_return, ep_len = self.train_env.reset(), False, 0., 0
        for _ in range(1, self.args.num_train_steps + 1):
            if self._global_step <= self.args.learning_starts:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.get_action(state)

            next_state, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_len += 1

            self.agent.env_buffer.push((state, action, reward, next_state, done))

            if len(self.agent.env_buffer) > self.args.batch_size and self._global_step > self.args.learning_starts:
                self.agent.update(self._global_step)

            if (self._global_step + 1) % self.args.eval_episode_interval == 0:
                self.eval()

            if self._global_step % self.args.save_snapshot_interval == 0:
                self.save_snapshot()

            self._global_step += 1
            if done:
                self._global_episode += 1
                print(f"Episode: {self._global_episode}, steps: {self._global_step}, return: {round(ep_return, 2)}")
                wandb.log({'episodic_return': ep_return, 'episodic_length': ep_len}, step=self._global_step)
                (state, info), done, ep_return, ep_len = self.train_env.reset(), False, 0., 0
            else:
                state = next_state

        self.train_env.close()

    def eval(self):
        steps, returns = 0, 0
        frames = []
        for e in range(self.args.num_eval_episodes):
            state, info = self.eval_env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = self.agent.get_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                returns += reward
                steps += 1
                state = next_state
                if self.args.record_eval and e == 0:
                    frame = self.eval_env.render()
                    if frame is not None:
                        frames.append(frame)

        avg_return = returns / self.args.num_eval_episodes
        wandb.log({'eval/return': avg_return, 'eval/length': steps / self.args.num_eval_episodes}, step=self._global_step)
        if self.args.record_eval and frames:
            self._save_video(frames, "eval")

    def _save_video(self, frames, tag):
        import imageio.v2 as imageio
        arr = np.stack(frames, axis=0).astype(np.uint8)
        out_path = Path(self.args.video_dir) / f"{tag}_step{self._global_step}.mp4"
        imageio.mimsave(out_path, arr, fps=self.args.video_fps, codec="libx264", format="ffmpeg")
        wandb.log({f"{tag}/video": wandb.Video(str(out_path), fps=self.args.video_fps, format="mp4")}, step=self._global_step)

    def save_snapshot(self, best=False):
        snapshot = self.checkpoint_path / ('best.pt' if best else f"{self._global_step}.pt")
        payload = {'agent': self.agent, '_global_step': self._global_step, '_global_episode': self._global_episode}
        torch.save(payload, snapshot)


def main():
    with open("highway.yaml", "r") as f:
        config = yaml.safe_load(f)
    args = parse_args(config['rpc_params'])

    with wandb.init(project='rpc-highway', entity='arjaras-university-of-pennsylvania',
                    group=args.env_name, config=args.__dict__):
        wandb.run.name = f"{args.env_name}_{args.seed}"
        workspace = HighwayWorkspace(args)
        workspace.train()


if __name__ == "__main__":
    main()
