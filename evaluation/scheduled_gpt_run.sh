#!/bin/bash
# Waits until 03:00 local (Pacific) time, checks the inputs, then runs GPT-3.5 and GPT-4 (pinned snapshots) on all
# three prompt styles into outputs/test_data_10_22/. Started under caffeinate so the Mac does not idle-sleep.
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
DATA=../dataset/test_data_10_22.csv
ONE_SHOT=../final_files_2026-10-16/one_shot_10_17_2026.json
TAG=test_data_10_22
LOG=logs/scheduled_gpt_$(date +%Y%m%d).log
mkdir -p logs

target=$(date -j -v3H -v0M -v0S +%s)
now=$(date +%s)
echo "$(date) waiting until $(date -r "$target") ($(( (target - now) / 60 )) min)" >> "$LOG"
[ "$target" -gt "$now" ] && sleep $((target - now))

echo "$(date) start" >> "$LOG"
export OPENAI_API_KEY="$(grep -E '^[[:space:]]*(export[[:space:]]+)?OPENAI_API_KEY=' ../.env | tail -1 | sed -E 's/^[[:space:]]*(export[[:space:]]+)?OPENAI_API_KEY=//; s/^["'"'"']//; s/["'"'"'][[:space:]]*$//')"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "$(date) STOPPED: no OPENAI_API_KEY in ../.env" >> "$LOG"; exit 1
fi
if [ ! -f "$DATA" ] || [ ! -f "$ONE_SHOT" ]; then
    echo "$(date) STOPPED: missing $DATA or $ONE_SHOT; nothing was sent to OpenAI" >> "$LOG"; exit 1
fi
if ! $PY check_inputs.py --data "$DATA" --one_shot "$ONE_SHOT" >> "$LOG" 2>&1; then
    echo "$(date) STOPPED: input check failed (see above); nothing was sent to OpenAI" >> "$LOG"; exit 1
fi

# run.py resumes from the rows already written, so each extra attempt only retries rows whose API call failed
for model in OpenAI/gpt-3.5-turbo-0125 OpenAI/gpt-4-0613; do
    (
        for attempt in 1 2 3; do
            $PY run.py --model "$model" --data "$DATA" --one_shot "$ONE_SHOT" --tag "$TAG" >> "logs/scheduled_${model#OpenAI/}.log" 2>&1
            if ! grep -q "rows failed" <(tail -n 20 "logs/scheduled_${model#OpenAI/}.log"); then break; fi
            echo "$(date) $model attempt $attempt left failed rows; retrying" >> "$LOG"
        done
        echo "$(date) $model finished" >> "$LOG"
        grep -E "^OpenAI/" "logs/scheduled_${model#OpenAI/}.log" | tail -n 3 | cut -c1-200 >> "$LOG"
    ) &
done
wait
echo "$(date) all done; outputs in outputs/$TAG/" >> "$LOG"
