#!/bin/bash
# Goal-conditioned Stop gate — the reusable version of the 2026-07-11 pilot.
#
# Inert unless the session opted in by writing .claude/SESSION_GOAL.md (first
# line = the goal; optional "max_continues: N" line). While that file exists,
# stopping is blocked with a reminder to keep working toward the goal — the
# agent ends the loop by DELETING the file when the goal is met or everything
# remaining is blocked (after writing the blockers list). A per-session counter
# fails open after max_continues (default 20) so a wedged session always ends.
INPUT=$(cat)
DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
GOAL_FILE="$DIR/.claude/SESSION_GOAL.md"
[ -f "$GOAL_FILE" ] || exit 0
command -v jq >/dev/null 2>&1 || exit 0

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"')
GOAL=$(head -1 "$GOAL_FILE")
MAX=$(grep -iE '^max_continues:' "$GOAL_FILE" | head -1 | grep -oE '[0-9]+' || true)
MAX=${MAX:-20}

COUNT_FILE="${TMPDIR:-/tmp}/claude-goal-gate-${SESSION_ID}.count"
COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
if [ "$COUNT" -ge "$MAX" ]; then
  # Fail open: the gate has re-prompted enough. Leave a trace and let it stop.
  echo "goal-gate: max_continues ($MAX) reached for session $SESSION_ID — allowing stop" >&2
  exit 0
fi
echo $((COUNT + 1)) > "$COUNT_FILE"

jq -n --arg goal "$GOAL" --arg n "$((COUNT + 1))" --arg max "$MAX" '{
  decision: "block",
  reason: ("Session goal is active (continue " + $n + "/" + $max + "): " + $goal + "\nKeep working toward it — self-score against the goal before claiming done. When the goal is met, or every remaining task is blocked (write the blockers list first), delete .claude/SESSION_GOAL.md and stop.")
}'
exit 0
