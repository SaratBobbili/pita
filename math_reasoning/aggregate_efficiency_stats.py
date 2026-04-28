import argparse
import json
import os

import numpy as np

from utils import read_jsonl


def summarize_array(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'var': 0.0,
            'min': 0.0,
            'max': 0.0,
            'p50': 0.0,
            'p90': 0.0,
            'p95': 0.0,
        }
    return {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'var': float(arr.var()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'p50': float(np.quantile(arr, 0.50)),
        'p90': float(np.quantile(arr, 0.90)),
        'p95': float(np.quantile(arr, 0.95)),
    }


def aggregate_hidden_norm_stats(rows, field_name):
    # rows[i][field_name] is a dict of model/split keys -> stats dict
    # each stats dict has count, mean, variance, min, max
    aggregated = {}
    for row in rows:
        row_stats = row.get(field_name, {})
        if not isinstance(row_stats, dict):
            continue
        for key, stats in row_stats.items():
            if not isinstance(stats, dict):
                continue
            count = float(stats.get('count', 0.0))
            if count <= 0:
                continue
            mean = float(stats['mean'])
            variance = float(stats['variance'])
            cur = aggregated.setdefault(key, {
                'count': 0.0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
            })
            cur['count'] += count
            cur['sum'] += count * mean
            cur['sum_sq'] += count * (variance + mean * mean)
            cur['min'] = min(cur['min'], float(stats['min']))
            cur['max'] = max(cur['max'], float(stats['max']))
    finalized = {}
    for key, cur in aggregated.items():
        if cur['count'] <= 0:
            continue
        mean = cur['sum'] / cur['count']
        variance = max(cur['sum_sq'] / cur['count'] - mean * mean, 0.0)
        finalized[key] = {
            'count': float(cur['count']),
            'mean': float(mean),
            'variance': float(variance),
            'std': float(np.sqrt(variance)),
            'min': float(cur['min']),
            'max': float(cur['max']),
        }
    return finalized


parser = argparse.ArgumentParser(description='Aggregate raw efficiency logs into final metrics.')
parser.add_argument('--efficiency_log_dir', required=True, type=str,
                    help='Path to efficiency log directory created by train/eval scripts.')
parser.add_argument('--output_path', default=None, type=str,
                    help='Output json path. Defaults to <efficiency_log_dir>/efficiency_report.json')
args = parser.parse_args()

if args.output_path is None:
    output_path = os.path.join(args.efficiency_log_dir, 'efficiency_report.json')
else:
    output_path = args.output_path

train_step_metrics_path = os.path.join(args.efficiency_log_dir, 'train_step_metrics.jsonl')
train_eval_metrics_path = os.path.join(args.efficiency_log_dir, 'train_eval_metrics.jsonl')
infer_example_metrics_path = os.path.join(args.efficiency_log_dir, 'infer_example_metrics.jsonl')
infer_batch_metrics_path = os.path.join(args.efficiency_log_dir, 'infer_batch_metrics.jsonl')
training_stats_path = os.path.join(os.path.dirname(args.efficiency_log_dir), 'training_stats.json')
inference_stats_path = os.path.join(args.efficiency_log_dir, 'inference_stats.json')

report = {}

if os.path.exists(train_step_metrics_path):
    train_steps = read_jsonl(train_step_metrics_path)
    report['training'] = {
        'num_logged_steps': len(train_steps),
        'step_wall_sec': summarize_array([x['step_wall_sec'] for x in train_steps]),
        'examples_per_sec': summarize_array([x['examples_per_sec'] for x in train_steps]),
        'tokens_per_sec': summarize_array([x['tokens_per_sec'] for x in train_steps]),
        'train_loss_accumulated': summarize_array([x['train_loss_accumulated'] for x in train_steps]),
        'gradient_norm': summarize_array([x['gradient_norm'] for x in train_steps]),
        'per_example_time_sec': summarize_array([
            x['step_wall_sec'] / max(x['num_examples'], 1e-12) for x in train_steps
        ]),
    }
    report['training_totals_from_steps'] = {
        'total_examples': float(sum(x['num_examples'] for x in train_steps)),
        'total_tokens': float(sum(x['num_tokens'] for x in train_steps)),
        'total_loss_tokens': float(sum(x['num_loss_tokens'] for x in train_steps)),
        'total_flops_trainable_est': float(sum(x['flops_trainable_est'] for x in train_steps)),
        'total_flops_total_est': float(sum(x['flops_total_est'] for x in train_steps)),
        'total_wall_sec': float(sum(x['step_wall_sec'] for x in train_steps)),
    }
    hidden_norms = aggregate_hidden_norm_stats(train_steps, 'hidden_norm_stats')
    if hidden_norms:
        report['training_hidden_norms_from_steps'] = hidden_norms

if os.path.exists(train_eval_metrics_path):
    train_eval = read_jsonl(train_eval_metrics_path)
    report['train_time_eval'] = {}
    for split in ['id', 'ood']:
        split_rows = [x for x in train_eval if x['eval_split'] == split]
        report['train_time_eval'][split] = {
            'num_logged_eval_events': len(split_rows),
            'eval_wall_sec': summarize_array([x['eval_wall_sec'] for x in split_rows]),
            'examples_per_sec': summarize_array([x['examples_per_sec'] for x in split_rows]),
            'tokens_per_sec': summarize_array([x['tokens_per_sec'] for x in split_rows]),
        }
    hidden_norms = aggregate_hidden_norm_stats(train_eval, 'hidden_norm_stats')
    if hidden_norms:
        report['train_time_eval_hidden_norms'] = hidden_norms

if os.path.exists(infer_example_metrics_path):
    infer_examples = read_jsonl(infer_example_metrics_path)
    report['inference'] = {
        'num_examples': len(infer_examples),
        'prompt_tokens': summarize_array([x['prompt_tokens'] for x in infer_examples]),
        'generated_tokens': summarize_array([x['generated_tokens'] for x in infer_examples]),
        'total_wall_sec_per_example': summarize_array([x['total_wall_sec_per_example'] for x in infer_examples]),
        'generate_wall_sec_per_example': summarize_array([x['generate_wall_sec_per_example'] for x in infer_examples]),
        'guidance_calls_per_example': summarize_array([x['guidance_calls_per_example'] for x in infer_examples]),
        'traj_kl': summarize_array([x['traj_kl'] for x in infer_examples]),
        'guidance_overhead_ratio_flops': summarize_array([x['guidance_overhead_ratio_flops'] for x in infer_examples]),
    }
    total_generated_tokens = float(sum(x['generated_tokens'] for x in infer_examples))
    total_wall = float(sum(x['total_wall_sec_per_example'] for x in infer_examples))
    total_ref_flops = float(sum(x['ref_only_flops_est'] for x in infer_examples))
    total_guided_flops = float(sum(x['guided_flops_est'] for x in infer_examples))
    report['inference_totals_from_examples'] = {
        'total_generated_tokens': total_generated_tokens,
        'total_wall_sec': total_wall,
        'tokens_per_sec': float(total_generated_tokens / max(total_wall, 1e-12)),
        'total_ref_only_flops_est': total_ref_flops,
        'total_guided_flops_est': total_guided_flops,
        'guidance_overhead_ratio_flops': float(total_guided_flops / max(total_ref_flops, 1e-12)),
    }

if os.path.exists(infer_batch_metrics_path):
    infer_batches = read_jsonl(infer_batch_metrics_path)
    hidden_norms = aggregate_hidden_norm_stats(infer_batches, 'hidden_norm_stats')
    if hidden_norms:
        report['inference_hidden_norms_from_batches'] = hidden_norms

if os.path.exists(training_stats_path):
    with open(training_stats_path, 'r') as f:
        report['training_stats'] = json.load(f)
if os.path.exists(inference_stats_path):
    with open(inference_stats_path, 'r') as f:
        report['inference_stats'] = json.load(f)

if 'training_stats' in report and 'inference_stats' in report:
    report['end_to_end'] = {
        'e2e_wall_clock_sec': float(
            report['training_stats']['wall_clock_time_sec'] + report['inference_stats']['inference_wall_time_sec']
        ),
        'e2e_flops_est': float(
            report['training_stats']['total_flops_total_est'] + report['inference_stats']['total_guided_flops_est']
        ),
        'train_wall_clock_sec': float(report['training_stats']['wall_clock_time_sec']),
        'inference_wall_clock_sec': float(report['inference_stats']['inference_wall_time_sec']),
    }

with open(output_path, 'w') as f:
    json.dump(report, f, indent=2)

print('Wrote efficiency report to', output_path)
