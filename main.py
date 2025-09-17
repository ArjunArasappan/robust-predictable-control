import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym


from buffer import ReplayBuffer
from models import Encoder, LatentDynamics, Policy, QFunction
from utils import soft_update
from train import compute_rpc_losses


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_obs(obs):
    """Ensure observation is a 1D float32 vector."""
    if isinstance(obs, dict):
        obs = obs.get("observation", obs)
    arr = np.array(obs, dtype=np.float32)
    return arr.reshape(-1)


def evaluate(env, encoder, policy, episodes: int, device: torch.device) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        s = flatten_obs(obs)
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                s_t = torch.from_numpy(s).to(device).unsqueeze(0)
                z, _, _, _ = encoder(s_t)
                a_cont, _ = policy(z)  # shape [1, action_dim]
                a_np = a_cont.squeeze(0).cpu().numpy()
            (obs, r, terminated, truncated, _) = env.step(a_np)
            done = terminated or truncated
            s = flatten_obs(obs)
            ep_r += float(r)
        total += ep_r
    return total / float(episodes)


def main():
    parser = argparse.ArgumentParser(description="RPC training on Walker2d-v5")
    parser.add_argument("--env", type=str, default="Walker2d-v5")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lambda-info", type=float, default=1.0, help="information penalty λ")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # ----- Environment -----
    env = gym.make(args.env)
    obs_sample, _ = env.reset()
    action_dim = env.action_space.shape[0]
    state_dim = flatten_obs(obs_sample).shape[0]

    # ----- Models -----
    encoder = Encoder(state_dim, args.latent_dim).to(device)
    dynamics = LatentDynamics(args.latent_dim, action_dim).to(device)
    policy = Policy(args.latent_dim, action_dim).to(device)
    q_func = QFunction(state_dim, action_dim).to(device)
    target_q = QFunction(state_dim, action_dim).to(device)
    target_q.load_state_dict(q_func.state_dict())

    # ----- Optimizers -----
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    opt_dyn = torch.optim.Adam(dynamics.parameters(), lr=args.lr)
    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_q = torch.optim.Adam(q_func.parameters(), lr=args.lr)

    # ----- Replay -----
    replay = ReplayBuffer(max_size=args.buffer_size)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ----- Training Loop -----
    global_step = 0
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        s = flatten_obs(obs)
        done = False
        ep_ret = 0.0
        steps = 0

        while not done and steps < args.max_steps:
            with torch.no_grad():
                s_t = torch.from_numpy(s).to(device).unsqueeze(0)
                z, _, _, _ = encoder(s_t)
                a_cont, _ = policy(z)  # [1, action_dim]
                a_np = a_cont.squeeze(0).cpu().numpy()

            next_obs, r, terminated, truncated, _ = env.step(a_np)
            done = terminated or truncated
            s_next = flatten_obs(next_obs)

            replay.push(s, a_np, float(r), s_next, float(done))

            s = s_next
            ep_ret += float(r)
            steps += 1
            global_step += 1

            # Learn
            if len(replay) >= args.batch:
                s_b, a_b, r_b, s_next_b, done_b = replay.sample(args.batch)
                s_b = s_b.to(device)
                a_b = a_b.to(device)
                r_b = r_b.to(device)
                s_next_b = s_next_b.to(device)
                done_b = done_b.to(device)

                pol_loss, q_loss, info_cost = compute_rpc_losses(
                    s_b, a_b, r_b, s_next_b, done_b,
                    encoder, dynamics, policy, q_func, target_q,
                    lambda_info=torch.tensor(args.lambda_info, device=device),
                    gamma=args.gamma,
                )

                # Update Q
                opt_q.zero_grad()
                q_loss.backward()
                opt_q.step()

                # Update encoder, dynamics, policy
                opt_enc.zero_grad()
                opt_dyn.zero_grad()
                opt_pol.zero_grad()
                pol_loss.backward()
                opt_enc.step()
                opt_dyn.step()
                opt_pol.step()

                # Target update
                soft_update(target_q, q_func, tau=args.tau)

        # Logging
        print(f"[Ep {ep:04d}] return={ep_ret:.2f} steps={steps} buffer={len(replay)}")

        # Periodic evaluation + checkpoint
        if ep % args.eval_every == 0:
            avg_r = evaluate(env, encoder, policy, args.eval_episodes, device)
            print(f"  -> eval_avg_return={avg_r:.2f}")
            ckpt_path = os.path.join(args.save_dir, f"rpc_walker2d_ep{ep}.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "dynamics": dynamics.state_dict(),
                "policy": policy.state_dict(),
                "q_func": q_func.state_dict(),
                "target_q": target_q.state_dict(),
                "cfg": vars(args),
                "state_dim": state_dim,
                "action_dim": action_dim,
            }, ckpt_path)
            print(f"  -> saved {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
