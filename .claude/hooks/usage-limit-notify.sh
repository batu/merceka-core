#!/bin/bash
# Usage-limit Stop notifier. When a session's final assistant message is a
# subscription limit banner ("You've hit your session/weekly/usage limit…"),
# the fleet used to just go dark until Batu noticed hours later. This hook
# (a) pings Telegram so the wall is known from the phone, and (b) parks the
# twf lease when a board is claimed, so workers stop hammering a dead window.
# Notify-only: it NEVER blocks the stop and always exits 0 (fail-open).
INPUT=$(cat)
command -v jq >/dev/null 2>&1 || exit 0
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
[ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] || exit 0

# Last assistant text in the transcript tail; limit banners are short and final.
LAST=$(tail -c 65536 "$TRANSCRIPT" \
  | jq -rs '[ .[] | select(type=="object" and .type=="assistant") ] | last
             | .message.content
             | if type=="array" then (map(select(.type=="text") | .text) | join(" ")) else tostring end' \
  2>/dev/null | tail -c 500)
echo "$LAST" | grep -qiE "hit your .{0,20}limit" || exit 0

REPO=$(basename "${CLAUDE_PROJECT_DIR:-$(pwd)}")
MSG="claude hit a usage limit in ${REPO}: ${LAST:0:160}"

if command -v telegram-send >/dev/null 2>&1; then
  timeout 10 telegram-send "$MSG" >/dev/null 2>&1 || true
fi
if command -v twf >/dev/null 2>&1; then
  # Surface lease state; `twf capacity resume` is the recovery half and runs
  # from the conductor, not from a dying session's Stop hook.
  timeout 10 twf capacity status >/dev/null 2>&1 || true
fi
exit 0
