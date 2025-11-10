from pathlib import Path
import argparse
from pathlib import Path
from types import SimpleNamespace
from eval_rpc import EvalMujocoWorkspace
import torch
import wandb
from argparse import Namespace

import os

def parse_run_metadata(run_dir):
    """
    Extracts env_name, seed, kl from folder name of the form:
    2025-11-03_02-17-17_RPC_Walker2d-v5_seed_85_kl_1_qk29e9et
    """
    name = run_dir
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
            hash = str(parts[i+2])
            
    print(env_name, seed, kl, hash)

    return {
        "env_name": env_name,
        "seed": seed,
        "kl": kl,
        "hash" : hash
    }



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint_name", type=str, default="best.pt",
                   help="Checkpoint filename inside each run dir")
    p.add_argument("--project", type=str, default="rpc-eval-2",
                   help="wandb project name for evals")
    p.add_argument("--entity", type=str, default=None,
                   help="wandb entity (optional)")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device to evaluate on")
    return p.parse_args()




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




def eval_single_run(run_dir, iter = 'best'):
    path = run_dir + f'/{iter}.pt'
    
    config = parse_run_metadata(run_dir)
    hash = config['hash']
    
    payload = torch.load(path)
    args = payload['args']
    
    if isinstance(args, dict):
        args = Namespace(**args)
        
    run_id = f'reval-{args.seed}-kl{args.kl_constraint}-{hash}'
    print(run_id)
    
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


        workspace = EvalMujocoWorkspace(load_path = run_dir, iter = iter)

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
        


def main():
    args = parse_args()
    path = '../robust-predictable-control/checkpoints/RPC_Walker2d-v5/_exp3/'
    runs_root = Path('../robust-predictable-control/checkpoints/RPC_Walker2d-v5/_exp3/')

    # iterate over subdirectories
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])

    print(f"Found {len(run_dirs)} run dirs in {runs_root}")
    for run_dir in run_dirs:
        
        eval_single_run(path + run_dir.name)
        
        
        

if __name__ == "__main__":
    main()
