#!/usr/bin/env python3
"""PreToolUse guard for PRESLEY: block hand-rolled process-wait loops.

Reads the Claude Code hook JSON on stdin. Denies Bash commands that poll for
a process to disappear (`until ! pgrep -f X; do sleep 60; done`, `while ps
-p $PID; do sleep; done`, `while kill -0 $PID; …`).

Why: the harness runs the command inside `bash -c "<the whole command
string>"`, and that string *contains* the pgrep pattern -- so `pgrep -f`
matches the watcher's own process and the loop can never terminate. The
watched job finishes, the watcher spins until timeout, and the completion
goes unnoticed. This has burned >1h of wall clock at least twice.

There is also nothing to poll for: `Bash` with `run_in_background: true`
re-invokes Claude when the process exits, and `Monitor` streams progress
events. Both are strictly better than any loop written here.

Narrow by construction -- a denial needs all three of a loop keyword, a
process-liveness check, and a sleep in the body. A bare `pgrep`, a bare
`sleep`, or a polling loop over something other than process liveness (a
file, an HTTP endpoint, a CI run) is allowed through.
"""
import json
import re
import sys

LOOP = re.compile(r"\b(until|while)\b")
SLEEP = re.compile(r"\bsleep\s+[\d.]+")
# Process-liveness probes -- the class of condition that can self-match.
PROBE = re.compile(
    r"\bpgrep\b|\bpidof\b|\bkill\s+-0\b|\bps\s+(-p|-e|aux|ax)\b|\bpkill\s+-0\b"
)

REASON = (
    "Blocked: this looks like a hand-rolled wait-for-process loop. It cannot "
    "work here -- the harness runs your command as `bash -c \"<whole command "
    "string>\"`, so the loop's own process matches its own pgrep/ps pattern "
    "and the condition never becomes true. The job finishes and the watcher "
    "spins until timeout.\n\n"
    "Use instead:\n"
    "  - Bash with run_in_background: true -- detaches, survives across "
    "turns, and re-invokes you on exit with the output-file path. No polling.\n"
    "  - Monitor -- if you want progress events during the run. Filter for "
    "failure signatures (Traceback|Error|Killed|OOM) too, not just success.\n"
    "  - Foreground Bash with an explicit timeout (max 600000 ms) if the job "
    "genuinely finishes in under 10 minutes.\n\n"
    "See the waiting rule in the global CLAUDE.md."
)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # Malformed input: stay out of the way.

    if payload.get("tool_name") != "Bash":
        sys.exit(0)

    command = payload.get("tool_input", {}).get("command", "")
    if not isinstance(command, str) or not command:
        sys.exit(0)

    if LOOP.search(command) and PROBE.search(command) and SLEEP.search(command):
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": REASON,
            }
        }))
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
