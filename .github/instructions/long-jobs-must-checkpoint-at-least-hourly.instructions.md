---
applyTo: "scripts/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Long jobs must checkpoint at least hourly' section. Copilot's cloud agent and code review
     read the whole of AGENTS.md; this copy is for Copilot Chat, which
     reads only .github/. -->

## Long jobs must checkpoint at least hourly

SSH to this host drops a couple of times a day, and restoration runs take
hours. Any job expected to exceed an hour checkpoints at least every 60
minutes of wall clock, independent of its epoch/step cadence, and its resume
path is verified *before* it is relied on. Long scripts append a progress line
(step, metric, timestamp) to their log at least every 10 minutes, so a silent
hang is visible in minutes and any progress watcher always has something fresh to match on. Launch detached — never attached to a shell an SSH drop takes with it.
