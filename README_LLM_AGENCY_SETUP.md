# Agency Setup Required

This file is a reminder for the next LLM session working in this project.
Delete it after completing the setup.

## Status: Not initialized

## What to do

This project needs agency setup:

1. Run `agency init` (scaffolds agents/, installs always-on skills, syncs)
2. Create vulture_whitelist.py — this is a library so vulture may flag
   public API functions as "unused" since nothing calls them internally
3. The pre-commit hook (ruff + vulture) will be installed automatically
4. No optional skills likely needed — this is a core library

## Reference

- Agency CLI: `/home/batu/Desktop/utolye/agency/`
- Skill catalog: `agency list-skills`
- Skill index: `/home/batu/Desktop/utolye/agency/src/agency/catalog/skills/INDEX.md`
- Global plugins (compound-engineering, mgrep, last30days): already installed at ~/.claude/
