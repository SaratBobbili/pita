# Star-graph experiments

This code is heavily inspired by [https://github.com/jinpz/q_sharp], the official code base for Zhou et al. 2025. The PITA training algorithm for the star graph reasoning is a fork of the Q-# classifier, and can be run by changing the classifier files.

## Requirements
The following packages are needed to run the code:
1. *torch* 2.2.0
2. *transformers* 4.37.2
3. *numpy* 1.26.3
4. *tqdm* 4.66.1
5. *wandb* 0.16.2

## Data
You can generate your own pre-training data with `python data/graphs.py` and your own pairwise post-training data with
> python collect_rollouts.py --model gpt2 \
    --batch_size 1024 --dataset graph --deg 2 --path 5 --num_nodes 50 \
    --model_ckpt <path_to_pretrained_model>

For reproducibility, here's the link to the original checkpoints of all the baseline algorithms and the datasets used [box link](https://cornell.box.com/s/ustn3954a5o3viohayvp6nr956kx8tfe). The data should be copied to `data/datasets/` and checkpoints should be copied to `checkpoints/` directory.

## Usage

To pre-train a GPT-2 model with standard next-token prediction on a star graph with degree 2 and path length 5 with 50 possible node values, run the command
> python train.py --dataset graph --deg 2 --path 5 --num_nodes 50 --batch_size 256 --lr 0.00025 --epochs 30 --eval_every 1 --save_every 2 --model gpt2 --weight_decay 0.1 --eval_train --wandb_entity <fill_in> --use_wandb --seed 1337

To train a GPT-2 classifier for CD or Q#, run the command
> python train_classifier.py --dataset graph --deg 2 --path 5 --num_nodes 50 --batch_size 256 --lr 0.00025 --epochs 10 --eval_every 1 --save_every 1 --model gpt2 --weight_decay 0.1 --wandb_entity <fill_in> --use_wandb --seed 1337 --compile

To train a GPT-2 classifier for PITA, run the command
> python train_classifier_pref.py --dataset graph --deg 2 --path 5 --num_nodes 50 --batch_size 256 --lr 0.00025 --epochs 10 --eval_every 1 --save_every 1 --model gpt2 --weight_decay 0.1 --wandb_entity <fill_in> --use_wandb --seed 1337 --compile

We can then use the classifier ckpt to run CD, Q# or PITA. Please see `scripts/evaluate.sh` for an example command.

To post-train a GPT-2 model with REINFORCE, run the command
> python train_reinforce.py --dataset graph --deg 2 --path 5 --num_nodes 50 --batch_size 256 --lr 1e-05 --epochs 10 --eval_every 1 --save_every 1 --piref_ckpt <path_to_pretrained_model> --model gpt2 --baseline 1 --weight_decay 0.1 --wandb_entity <fill_in> --use_wandb --seed 1337 --compile

If you've downloaded our checkpoints, <path_to_pretrained_model> can be replaced with `checkpoints/pretrain/deg_2_path_5.pt`.

To post-train a GPT-2 model with RPO, run the command
> python train_dpo.py --dataset graph --deg 2 --path 5 --num_nodes 50 --batch_size 256 --lr 0.0001 --epochs 10 --eval_every 1 --save_every 1 --piref_ckpt <path_to_pretrained_model> --model gpt2 --use_rpo 1 --weight_decay 0.1 --wandb_entity <fill_in> --use_wandb --seed 1337 --compile

Remove the `--use_rpo` flag for DPO.
