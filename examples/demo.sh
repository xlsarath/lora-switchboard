#!/usr/bin/env bash
# End-to-end demo: deploy -> smoke pass -> promote -> bad deploy blocked -> rollback.
#
# Against docker compose (default):   ./examples/demo.sh
# Against a local `lora-switchboard serve`:
#   LSB_CLI="lora-switchboard" SMOKE=examples/smoke-support-bot.yaml ./examples/demo.sh
set -euo pipefail

LSB_CLI=${LSB_CLI:-"docker compose exec -T switchboard lora-switchboard"}
SMOKE=${SMOKE:-/app/examples/smoke-support-bot.yaml}
URL=${URL:-http://127.0.0.1:8000}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
STAMP=$(date +%s)   # fresh adapter paths per run, so the demo is re-runnable

lsb() {
  local shown=() arg
  for arg in "$@"; do [[ $arg == *" "* ]] && shown+=("\"$arg\"") || shown+=("$arg"); done
  echo; echo "\$ lora-switchboard ${shown[*]}"
  $LSB_CLI "$@"
}
ask() {
  echo
  echo "\$ curl -s $URL/v1/chat/completions -d '{\"model\": \"support-bot\", ...}' | jq -r '.choices[0].message.content'"
  curl -sf "$URL/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model": "support-bot", "messages": [{"role": "user", "content": "Where is my order?"}]}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

lsb deploy support-bot --path "/adapters/support-bot/$STAMP-a" --base-model "$BASE_MODEL" \
    --alias support-bot --smoke "$SMOKE" --notes "first release"
ask
lsb deploy support-bot --path "/adapters/support-bot/$STAMP-b" \
    --alias support-bot --smoke "$SMOKE" --notes "more training data"
ask
lsb deploy support-bot --path "/adapters/support-bot/$STAMP-c-broken" \
    --alias support-bot --smoke "$SMOKE" --notes "bad learning rate" || true
ask
lsb rollback support-bot
ask
lsb status
lsb history --adapter support-bot --limit 12
