#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

if echo "$COMMAND" | grep -qE 'git\s+push\s+.*--force|git\s+reset\s+--hard|git\s+checkout\s+\.|git\s+clean\s+-f'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "ask",
      permissionDecisionReason: "Destructive git command detected (force push, hard reset, or clean). Requires user approval."
    }
  }'
  exit 0
fi

exit 0
