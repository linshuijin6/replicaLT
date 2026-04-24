# Workspace Agents

- Follow the project-specific rules in `.agent/rules.md`.
- Treat the current workspace root as the base directory for all `sync/` files.
- **Mandatory completion gate:** for every conversation, append a concise record to `sync/chat-log.md` before ending the turn.
- **Mandatory completion gate on file changes:** if any file is created, edited, or deleted in the turn, append `sync/changes-log.md` before ending the turn.
- **Long-task anti-skip rule:** do not postpone logging until “maybe later”. For multi-step or long-running tasks, write the required logs after the main edits are done and before the final summary.
- **Do not declare completion early:** if the required log steps for the turn are not finished, the task is not complete and you must not output the final answer yet.
- When code changes happened, attempt the Notion diary sync workflow after local logs are written; if Notion tooling is unavailable or fails, keep local logging mandatory and only mention the failure briefly at the end.
- If the user asks to continue a previous topic, first read the latest entries in `sync/chat-log.md` to recover context.
- Canonical user-level source copy for these sync rules: `/home/data/linshuijin/.config/sync-logger.instructions.md`.
