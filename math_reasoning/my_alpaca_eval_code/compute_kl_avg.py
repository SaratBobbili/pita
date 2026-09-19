# avg_traj_kl.py      run:  python compute_kl_avg.py file.jsonl
import json, sys

total, count = 0.0, 0

with open(sys.argv[1]) as f:
    for line in f:
        kl_list = json.loads(line)["traj_kl"]  # always a list
        total += sum(kl_list)
        count += len(kl_list)

print(total / count)
