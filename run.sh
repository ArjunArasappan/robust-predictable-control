#!/usr/bin/env bash
# sweep_kl_seed_max3.sh
set -euo pipefail

# ----- Search spaces -----
KLS=(0.1 0.4 1.6 6.4 25.6)
SEEDS=(43) #(94 75 43) # add more seeds as needed

# ----- Config -----
OUTDIR="${OUTDIR:-runs}"          # override: OUTDIR=rpc_runs ./sweep...
MAX_JOBS="${MAX_JOBS:-3}"         # change concurrency with MAX_JOBS=N
BASE_CMD=(python -u train.py)     # -u = unbuffered for live logs
EXTRA_ARGS=("$@")                 # forwarded to each run

mkdir -p "$OUTDIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }

# Detect wait -n support properly
have_wait_n=0
if help wait >/dev/null 2>&1 && help wait | grep -q -- " -n "; then
  have_wait_n=1
fi

throttle() {
  # Ensure we don't exceed MAX_JOBS concurrent background tasks
  while :; do
    running=$(jobs -pr | wc -l | tr -d ' ')
    if (( running < MAX_JOBS )); then
      break
    fi
    if (( have_wait_n )); then
      wait -n || true
    else
      sleep 0.5
    fi
  done
}

launch() {
  local kl="$1" seed="$2"
  local run_name="kl=${kl}_seed=${seed}_$(timestamp)"
  local run_dir="${OUTDIR}/${run_name}"
  mkdir -p "$run_dir"

  # Optional: separate env per run (e.g., W&B)
  # export WANDB_RUN_ID="$run_name"

  echo ">>> Launching ${run_name}"
  {
    printf "Command: %s " "${BASE_CMD[@]}"
    printf "%s " "${EXTRA_ARGS[@]}"
    printf -- "--kl_constraint %s --seed %s\n\n" "$kl" "$seed"

    # Persist args for provenance
    printf -- "kl=%s\nseed=%s\nextra_args=%q\n" "$kl" "$seed" "${EXTRA_ARGS[*]}" > "${run_dir}/args.txt"

    # Execute the run
    "${BASE_CMD[@]}" \
      --kl_constraint "$kl" \
      --seed "$seed" \
      "${EXTRA_ARGS[@]}"
  } &> >(tee "${run_dir}/stdout_stderr.log")
}

# ----- Launch all combos with concurrency cap -----
for kl in "${KLS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    throttle
    launch "$kl" "$seed" &
  done
done

# Drain remaining jobs
wait
echo "All runs complete."
