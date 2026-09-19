# Evaluation-phase efficiency: PITA / Q# / CD vs reference model

Llama-3-8B-Instruct on GSM8K test (1319 problems), temp 0.8, top_p 0.9, seed 47.
PITA / Q# / CD total rows are the Table 9 values divided by 8 (one-rollout-per-problem budget). Per-example rows are unchanged.
Ref@1 / Ref@2 are unguided reference-model runs (`eval_ckpt.py --eta 0`) with 1 and 2 rollouts per problem, reported as-is.

| Metric | PITA | Q# | CD | Ref@1 | Ref@2 |
|---|---|---|---|---|---|
| Rollouts per problem | 1 | 1 | 1 | 1 | 2 |
| Overall inference wall-clock time (s) | 1,286.25 | 1,247.39 | 1,306.50 | 726.26 | 1,420.11 |
| Overall inference cost (GPU-hours) | 0.36 | 0.35 | 0.36 | 0.20 | 0.39 |
| Total generated tokens | 194,980 | 188,927 | 189,687 | 188,424 | 378,659 |
| End-to-end latency per example (s) | 0.9753 ± 0.4050 | 0.9457 ± 0.3889 | 0.9905 ± 0.4644 | 0.5506 ± 0.1667 | 0.5383 ± 0.2032 |
| Generation-only latency per example (s) | 0.9022 ± 0.3649 | 0.8758 ± 0.3535 | 0.9201 ± 0.4218 | 0.4776 ± 0.1427 | 0.4651 ± 0.1691 |
| Generated tokens per example | 147.82 ± 67.37 | 143.23 ± 65.96 | 143.81 ± 70.06 | 142.85 ± 62.44 | 143.54 ± 65.61 |
| Run-level throughput (tokens/s) | 151.58 | 151.46 | 145.18 | 259.45 | 266.64 |
| Guidance calls per example | 31.60 ± 11.85 | 30.61 ± 11.45 | 31.37 ± 13.34 | 0 | 0 |
| Reference-only compute estimate (FLOPs) | 3.131 × 10^15 | 3.034 × 10^15 | 3.046 × 10^15 | 3.026 × 10^15 | 6.081 × 10^15 |
| Guided compute estimate (FLOPs) | 1.906 × 10^16 | 1.798 × 10^16 | 1.855 × 10^16 | 3.026 × 10^15 | 6.081 × 10^15 |

## Accuracy and hypothesis check

| Method | Accuracy |
|---|---|
| Ref pass@1 | 0.5845 |
| CD pass@1 | 0.6181 |
| Q# pass@1 | 0.6403 |
| PITA pass@1 | 0.6929 |
| Ref pass@2 | 0.7475 |

- Accuracy: Ref pass@1 (0.5845) < PITA pass@1 (0.6929) < Ref pass@2 (0.7475).
- Compute (one-sample-equivalent wall-clock): Ref@1 (726.26 s) < PITA (1,286.25 s) < Ref@2 (1,420.11 s).

Two unguided rollouts beat one PITA-guided rollout on pass rate, but PITA's single-rollout accuracy comes at less compute than the two-rollout reference budget.

Sources: `checkpoints/llama_3_8b_instruct_gsm8k/ref_pass{1,2}/` (reward_stats + efficiency reports) and `checkpoints/llama_3_8b_instruct_gsm8k/training_costs/{pita,q_sharp,cd}/ckpt_10000/efficiency/`. Regenerate the normalized comparison with `python print_eval_efficiency_table.py`.
