#!/bin/bash
OUTPUT_DIR="/data/houwanlong/finllm-mi/outputs/sae"
REPORT="/tmp/exp_logs/status_report.txt"

echo "=== Experiment Status $(date) ===" > "$REPORT"

for i in 1 2 3 4 5 6 7; do
    SESSION="exp${i}"
    echo "" >> "$REPORT"
    echo "--- Experiment ${i} ---" >> "$REPORT"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Status: RUNNING" >> "$REPORT"
        echo "Last output:" >> "$REPORT"
        tmux capture-pane -t "$SESSION" -p 2>/dev/null | tail -10 >> "$REPORT"
    else
        echo "Status: FINISHED" >> "$REPORT"
        JSON_FILE=$(ls -t "${OUTPUT_DIR}/exp${i}_"*.json 2>/dev/null | head -1)
        if [ -n "$JSON_FILE" ]; then
            echo "Output: $JSON_FILE" >> "$REPORT"
            python3 -c "
import json
d = json.load(open('${JSON_FILE}'))
print('Experiment:', d.get('experiment', 'unknown'))
for k,v in d.items():
    if k not in ('detail','results','per_family','layer_scores','selectivity_matrix') and not isinstance(v, (list,dict)):
        print(f'  {k}: {v}')
if 'per_family' in d:
    print('Per-family:')
    for f,s in d['per_family'].items():
        sstr = str(s)[:120]
        print(f'  {f}: {sstr}')
" 2>&1 >> "$REPORT"
        else
            echo "No output file found" >> "$REPORT"
        fi

        LOG="/tmp/exp_logs/exp${i}.log"
        if [ -f "$LOG" ]; then
            echo "Last log lines:" >> "$REPORT"
            tail -5 "$LOG" >> "$REPORT"
        fi
    fi
done

echo "" >> "$REPORT"
echo "=== End Report ===" >> "$REPORT"
cat "$REPORT"
