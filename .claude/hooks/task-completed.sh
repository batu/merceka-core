#!/bin/bash
INPUT=$(cat)
TASK_SUBJECT=$(echo "$INPUT" | jq -r '.task_subject')

# Check if there are uncommitted changes (indicating code work)
if git diff --quiet HEAD 2>/dev/null && git diff --cached --quiet HEAD 2>/dev/null; then
  exit 0
fi

jq -n --arg task "$TASK_SUBJECT" '{
  hookSpecificOutput: {
    hookEventName: "TaskCompleted",
    additionalContext: ("REMINDER: Task \"" + $task + "\" completed with uncommitted code changes. Run typecheck before committing: npx tsc --noEmit")
  }
}'
exit 0
