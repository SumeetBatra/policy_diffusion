#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00
#SBATCH -N1
#SBATCH --output=tmp/train_vae_%j.log

eval "$(conda shell.bash hook)"
conda activate policy_diffusion

export XLA_PYTHON_CLIENT_PREALLOCATE=false

set -- 1111 2222 # seeds

for seed in "$@";
  do echo "Running seed $item";
  RUN_NAME="my_vae_experiment_seed_"$seed
  echo $RUN_NAME
  srun -c12 python -m scripts.train --experiment_spec_file=specs/train_vae.yaml \
                                    --results_dir=./results \
                                    # all -o are spec overrides to default params in the yaml
                                    -o=spec.name=$RUN_NAME \
                                    -o=spec.trainer.config.random_seed=$seed \
                                    -o=spec.grad_clip=false \
done
