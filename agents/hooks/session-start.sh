#!/bin/bash
INPUT=$(cat)
SOURCE=$(echo "$INPUT" | jq -r '.source')

# Self-heal the origin fetch refspec. Single-branch operations (e.g. a review
# bot running `git remote set-branches origin <branch>`) can narrow the shared
# .git/config and silently break `main` fetching for every worktree. Re-assert
# the standard refspec, quietly, on every session start.
STD='+refs/heads/*:refs/remotes/origin/*'
if git rev-parse --git-dir >/dev/null 2>&1 \
   && [ "$(git config --get-all remote.origin.fetch 2>/dev/null)" != "$STD" ]; then
  git config remote.origin.fetch "$STD" >/dev/null 2>&1
fi

if [ "$SOURCE" = "startup" ]; then
  CONTEXT="REMINDER: Set your tmux pane title with: pane-title <agent> \"<description>\" once you understand the task."

  # Surface README_LLM files if they exist
  FOUND_README=""
  for readme in README_LLM*.md README_LLM*.txt; do
    [ -f "$readme" ] || continue
    FOUND_README="$readme"
    break
  done

  if [ -n "$FOUND_README" ]; then
    CONTEXT="$CONTEXT\n\nACTION REQUIRED: Read $FOUND_README — it contains setup instructions for this project that should be completed before starting work."
  fi

  # Notify (never mutate) when agency-managed templates are behind the catalog.
  if command -v agency >/dev/null 2>&1; then
    FRESH=$(timeout 5 agency fresh-check 2>/dev/null | head -n1)
    if [ -n "$FRESH" ]; then
      CONTEXT="$CONTEXT\n\n$FRESH"
    fi
  fi

  jq -n --arg ctx "$CONTEXT" '{
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: $ctx
    }
  }'
fi
exit 0
