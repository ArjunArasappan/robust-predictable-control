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
from argparse import Namespace

import imageio.v2 as imageio
import numpy as np
import datetime
from pathlib import Path
import wandb


import cv2
import numpy as np
import torch

import cv2
import numpy as np

def draw_metrics_bar(frame, step, mode_text, metrics_text="", is_open_loop=None):
    img = frame.copy()
    h, w = img.shape[:2]

    # --- White top bar for metrics ---
    top_bar_h = max(24, h // 20)
    cv2.rectangle(img, (0, 0), (w, top_bar_h), (255, 255, 255), -1)
    font, scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 12
    cv2.putText(img, metrics_text, (margin, int(top_bar_h * 0.7)), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # --- Semi-transparent bottom bar for step/mode ---
    bottom_bar_h = max(24, h // 20)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bottom_bar_h), (w, h), (0, 0, 0), -1)
    img[h - bottom_bar_h:h] = cv2.addWeighted(overlay[h - bottom_bar_h:h], 0.4, img[h - bottom_bar_h:h], 0.6, 0)

    # --- Step text (left) ---
    cv2.putText(img, f"Step {step}", (margin, h - bottom_bar_h // 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # --- Mode text (right) ---
    color = (200, 200, 200)
    tw, _ = cv2.getTextSize(f"{mode_text}", font, scale, thickness)[0]
    cv2.putText(img, f"{mode_text}", (w - tw - margin, h - bottom_bar_h // 3), font, scale, color, thickness, cv2.LINE_AA)

    return img


def parse_run_metadata(run_dir: Path):
    """
    Extracts env_name, seed, kl from folder name of the form:
    2025-11-03_02-17-17_RPC_Walker2d-v5_seed_85_kl_1_qk29e9et
    """
    name = run_dir.name
    parts = name.split('_')

    env_name = None
    seed = None
    kl = None

    for i, p in enumerate(parts):
        if p.startswith("RPC") or p.startswith("RRPC"):
            # p = RPC_Walker2d-v5
            env_name = p.split("RPC_")[-1]
        if p == "seed":
            seed = int(parts[i+1])
        if p == "kl":
            kl = float(parts[i+1])
            hash = str(parts[i + 2])

    return {
        "env_name": env_name,
        "seed": seed,
        "kl": kl,
        "hash" : hash
    }





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

class EvalMujocoWorkspace:
    def __init__(self, load_path, iter):
        
        self.load_snapshot(load_path, iter)
        self.work_dir = Path.cwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.device=='cuda' else "cpu")

        self.setup()
        self.agent = make_agent(self.train_env, self.device, self.args)
                    
        self.agent.encoder.load_state_dict(self.encoder)
        self.agent.actor.load_state_dict(self.actor)
        self.agent.model.load_state_dict(self.model)
        
        self.set_seeds_everywhere()
        self._best_eval_returns = -np.inf
            
        
    def override_args(self):
        
        if isinstance(self.args, dict):
            self.args = Namespace(**self.args)
            
        self.args.video_dir = 'eval_videos'
        Path(self.args.video_dir ).mkdir(parents=True, exist_ok=True)
        
        self.kl_constraint = self._kl_target_bits()
        
    def load_snapshot(self, path, iter, best=False):
        if best:
            snapshot = path / 'best.pt'
        else:
            snapshot = path / Path(str(iter)+'.pt')
            
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
            

            
        self.override_args()

    def setup(self):
        
        self.train_env  = gym.make(self.args.env_name)  
        self.eval_env   = gym.make(self.args.env_name, render_mode="rgb_array")
        self.robust_env = gym.make(self.args.env_name, render_mode="rgb_array")
        self.checkpoint_path = os.path.join(
                self.work_dir, 'eval_checkpoints',
                f"{self.args.agent}_{self.args.env_name}",
                wandb.run.id
                )
        os.makedirs(self.checkpoint_path, exist_ok=True)

        
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
                
                

    def save_snapshot(self, best=False):
        keys_to_save = ['agent', '_global_step', '_global_episode', 'args']
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._global_step)+'.pt')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)




            
    def _save_and_log_video(self, frames, tag, eval_key):
        if not frames:
            return



        arr = np.stack(frames, axis=0).astype(np.uint8)
                
        kl_tag  = f"kl{self.kl_constraint}" if self.kl_constraint is not None else "klNA"

        run_id = wandb.run.id if wandb.run is not None \
                else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        env_dir = Path(self.args.video_dir) / self.args.env_name 
        env_dir.mkdir(parents=True, exist_ok=True)

        video_subdir = env_dir / f"{self.args.agent}_{self.args.seed}_{kl_tag}_{run_id}" / eval_key
        video_subdir.mkdir(parents=True, exist_ok=True)

        self._dump_args_if_missing(video_subdir)

        out_path = video_subdir / f"{tag}_step{self._global_step}_{kl_tag}.mp4"

        imageio.mimsave(
            out_path,
            arr,
            fps=int(self.args.video_fps),
            codec="libx264",
            format="ffmpeg"
        )





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

        total_kl = 0.0
        kl_log = []

        for e in range(self.args.num_eval_episodes):
            done = False
            state, info = self.eval_env.reset()

            if record_first and e == 0:
                frame = self.eval_env.render()   # (H,W,C) uint8
                if frame is not None:
                    metrics_text = f"EVAL targ_kl: {self.kl_constraint}, actual_kl = -"
                    annotated = draw_metrics_bar(
                        frame,
                        step=0,
                        mode_text="CLOSED-LOOP",
                        metrics_text=metrics_text,
                        is_open_loop=False,
                    )
                    frames.append(annotated)

            ep_steps = 0

            while not done:
                with torch.no_grad():
                    action, z = self.agent.get_action(state, True)
             

                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)

                exec_action = torch.tensor(action, device=self.device).unsqueeze(dim=0)
                with torch.no_grad():
                    actual_kl = self.agent.get_kl(state, next_state, exec_action)

                kl_val = float(actual_kl)
                kl_log.append(kl_val)
                

                metrics_text = (
                    f"targ_kl: {self.kl_constraint}, "
                    f"actual_kl = {round(kl_val, 2)}"
                )

                returns += float(reward)
                steps += 1
                ep_steps += 1
                state = next_state

                if record_first and e == 0:
                    frame = self.eval_env.render()
                    if frame is not None:
                        annotated = draw_metrics_bar(
                            frame,
                            step=ep_steps,
                            mode_text="CLOSED",
                            metrics_text=metrics_text,
                            is_open_loop=False,
                        )
                        frames.append(annotated)

        avg_return = returns / self.args.num_eval_episodes
        if avg_return >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = avg_return

        if record_first:
            self._save_and_log_video(frames, tag="eval", eval_key="eval")

        avg_kl = sum(kl_log) / len(kl_log)

        metrics = {
            "episodic_return": avg_return,
            "episodic_length": steps / self.args.num_eval_episodes,
            "avg_kl": avg_kl,
            "kl_log": kl_log
        }

        return metrics

            
    def eval_open_loop(self, p, blind = False):
        steps, returns = 0, 0.0
        record_first = self.args.record_eval

        frames = []
        robustness_p = p

        kl_log = []
        open_log = []

        metrics_text = f"p: {p}, blind {blind}, kl: {self.kl_constraint}, -"

        for e in range(self.args.num_eval_episodes):
            done = False
            state, info = self.eval_env.reset()

            last_z_dist = None
            last_action = None

            if record_first and e == 0:
                frame = self.eval_env.render()  # (H,W,C) uint8
                if frame is not None:
                    annotated = draw_metrics_bar(
                        frame,
                        step=0,
                        mode_text="N/A",
                        metrics_text=metrics_text,
                        is_open_loop=None,
                    )
                    frames.append(annotated)

            ep_steps = 0

            while not done:
                with torch.no_grad():

                    if (last_z_dist is not None) and (np.random.uniform(0, 1) < robustness_p):
                        action, predicted_z = self.agent.get_action_open_loop(last_z_dist, last_action)

                        if not blind:
                            last_action = torch.tensor(action, device=self.device).unsqueeze(dim=0)
                            last_z_dist = predicted_z

                        is_open = True
                    else:
                        action, z = self.agent.get_action(state, True)
                        last_action = torch.tensor(action, device=self.device).unsqueeze(dim=0)
                        last_z_dist = z
                        is_open = False

                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                exec_action = torch.tensor(action, device=self.device).unsqueeze(dim=0)

                with torch.no_grad():
                    actual_kl = self.agent.get_kl(state, next_state, exec_action)

                metrics_text = f"p: {p}, blind {blind}, kl: {self.kl_constraint}, {round(actual_kl, 2)}"

                done = bool(terminated or truncated)

                returns += float(reward)
                steps += 1
                ep_steps += 1
                state = next_state

                if record_first and e == 0:
                    kl_log.append(actual_kl)
                    open_log.append(is_open)

                    frame = self.eval_env.render()
                    if frame is not None:
                        mode_text = "OPEN" if is_open else "CLOSED"
                        annotated = draw_metrics_bar(
                            frame,
                            step=ep_steps,           # step count within the episode
                            mode_text=mode_text,
                            metrics_text=metrics_text,
                            is_open_loop=is_open,
                        )
                        frames.append(annotated)

        avg_return = returns / self.args.num_eval_episodes
        if avg_return >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = avg_return

        if record_first:
            tag = f"open_{robustness_p}_{blind}"
            self._save_and_log_video(frames, tag=tag, eval_key=tag)

        metrics = {
            "episodic_return": avg_return,
            "target_kl": self.kl_constraint,
            "episodic_length": steps / self.args.num_eval_episodes,
            
            "kl_log": kl_log,
            "avg_kl": sum(kl_log) / len(kl_log),
            "open_hist": open_log,
        }

        return metrics
     
            
    def eval_open_loop_freq(self, rate, blind = False): # rate is number of open loop evals per observation: 0 -> open loop, 1 -> p = 0.5, 0.5 -> 1 open loop, 2 observation, 3 -> 3 open loop, 1 observation
        steps, returns = 0, 0.0
        record_first = getattr(self.args, "record_eval", False)

        frames = []
        kl_log = []
        open_log = []

        for e in range(self.args.num_eval_episodes):
            done = False
            state, info = self.eval_env.reset()

            last_z = None
            last_action = None

            if record_first and e == 0:
                frame = self.eval_env.render()  # (H,W,C) uint8
                if frame is not None:
                    annotated = draw_metrics_bar(
                        frame,
                        step=0,
                        mode_text="N/A",
                        metrics_text="",
                        is_open_loop=None,
                    )
                    frames.append(annotated)

            ep_steps = 0

            while not done:
                with torch.no_grad():
                    observe_now = (rate <= 0) or (ep_steps % (rate + 1) == 0) or (last_z is None)

                    if observe_now:
                        action, z = self.agent.get_action(state, True)
                        last_z = z
                        last_action = torch.as_tensor(action, device=self.device).unsqueeze(0)
                        is_open = False
                    else:
                        action, predicted_z = self.agent.get_action_open_loop(last_z, last_action)

                        if not blind:
                            last_z = predicted_z
                            last_action = torch.as_tensor(action, device=self.device).unsqueeze(0)

                        is_open = True

                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)

                exec_action = torch.tensor(action, device=self.device).unsqueeze(dim=0)
                with torch.no_grad():
                    actual_kl = self.agent.get_kl(state, next_state, exec_action)

                metrics_text = (
                    f"rate: {rate}, blind {blind}, "
                    f"kl: {self.kl_constraint}, {round(actual_kl, 2)}"
                )

                returns += float(reward)
                steps += 1
                ep_steps += 1
                state = next_state

                if record_first and e == 0:
                    # store time-series history (first episode only)
                    kl_log.append(float(actual_kl))
                    open_log.append(bool(is_open))

                    frame = self.eval_env.render()
                    if frame is not None:
                        mode_text = "OPEN-LOOP" if is_open else "CLOSED-LOOP"
                        annotated = draw_metrics_bar(
                            frame,
                            step=ep_steps,
                            mode_text=mode_text,
                            metrics_text=metrics_text,
                            is_open_loop=is_open,
                        )
                        frames.append(annotated)

        avg_return = returns / self.args.num_eval_episodes
        if avg_return >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = avg_return

        if record_first:
            tag = f"open_freq_{rate}_{blind}"
            self._save_and_log_video(
                frames,
                tag=tag,
                eval_key=tag,
            )

        metrics = {
            "episodic_return": avg_return,
            "target_kl": self.kl_constraint,
            "episodic_length": steps / self.args.num_eval_episodes,
            "rate": rate,
            
            "kl_log": kl_log,
            "avg_kl": sum(kl_log) / len(kl_log),
            "open_log": open_log,
        }

        return metrics


        
            
            


def log_eval(metrics):
    """Closed-loop baseline eval (no open-loop)."""
    open_frac = 0.0  # fully closed-loop

    log_dict = {
        "grid/episodic_return": metrics.get("episodic_return"),
        "grid/episodic_length": metrics.get("episodic_length"),
        "grid/target_kl":       metrics.get("target_kl"),

        "grid/schedule_type": "eval",  # closed-loop baseline
    }

    wandb.log(log_dict)


def log_open_metrics(metrics, p, blind=False):
    """Probabilistic open-loop schedule: eval_open_loop(p)."""
    # here p itself is the open-loop fraction (in expectation)
    open_frac = float(p)

    log_dict = {
        "grid/episodic_return": metrics.get("episodic_return"),
        "grid/episodic_length": metrics.get("episodic_length"),
        "grid/target_kl":       metrics.get("target_kl"),
        "grid/p": float(p),
        
        "grid/schedule_type": "prob",     # probabilistic schedule
        "grid/open_frac": open_frac,
        "grid/blind": int(blind),
    }

    wandb.log(log_dict)


def log_open_freq_metrics(metrics, rate, blind=False):
    """Periodic open-loop schedule: eval_open_loop_freq(rate)."""
    # effective open-loop fraction ~ rate / (rate + 1)
    open_frac = rate / (rate + 1) if rate >= 0 else 0.0

    log_dict = {
        "grid/episodic_return": metrics.get("episodic_return"),
        "grid/episodic_length": metrics.get("episodic_length"),
        "grid/target_kl":       metrics.get("target_kl"),
        "grid/rate":            rate,

        "grid/schedule_type": "freq",     # frequency-based schedule
        "grid/open_frac": open_frac,
        "grid/blind": int(blind),
    }

    wandb.log(log_dict)


    
    

def main():
    

    
    parent_dir = '../robust-predictable-control/checkpoints/RPC_Walker2d-v5/_exp3/'
    load_path1 = parent_dir + '2025-11-06_15-37-54_RPC_Walker2d-v5_seed_58_kl_1_y435sbqz'
    load_path60 = parent_dir + '2025-11-06_21-54-48_RPC_Walker2d-v5_seed_58_kl_60_8z83t41b'
    load_path3 = parent_dir + '2025-11-06_15-37-54_RPC_Walker2d-v5_seed_58_kl_0.3_c5t77u8k'
    load_path3=load_path60
    
    payload = torch.load(load_path3 + '/1000000.pt')
    args = payload['args']
    
    if isinstance(args, dict):
        args = Namespace(**args)


    


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


        iter = 1_000_000
        workspace = EvalMujocoWorkspace(load_path = load_path3, iter = iter)

        metrics = workspace.eval()
        log_eval(metrics)
        
        p_values = [
            i / 10 for i in range(1, 10)
        ]

            
        for blind in [False, True]:
            for p in p_values:
                rate = int(1 / p) - 1
                
                m_open = workspace.eval_open_loop(p=p, blind=blind)
                log_open_metrics(m_open, p=p, blind=blind)

                m_freq = workspace.eval_open_loop_freq(rate=rate, blind=blind)
                log_open_freq_metrics(m_freq, rate=rate, blind=blind)



if __name__ == '__main__':
    main()