# launchd Migration Plan — A-share Production Scheduler

Author: cx | Date: 2026-06-16 | Status: design only, no code changes

## 1. Why

macOS treats `crontab` as legacy. `crontab -` (what
`install_crontab.py --apply` runs) triggers a Full Disk Access
dialog that hangs unattended cx automation. `launchd` is native,
TCC-integrated, and gives per-job logs and structured state via
`launchctl print`.

## 2. What we're migrating

`install_crontab.py::managed_jobs()` emits ~50 entries (4
market-hours, ~25 post-close LLM/policy/cache, ~15 shadow
overlays + monitoring, 6 weekly Saturday). Field mapping:

| CronJob field | launchd target |
|---|---|
| `schedule` (5-field cron) | `StartCalendarInterval` array of dicts |
| `target` (argv) | `ProgramArguments` (inside existing wrapper chain) |
| `env_vars` tuple | `EnvironmentVariables` dict |
| `network` profile | unchanged; stays inside `run_network_job.py` |
| `timeout_sec` | unchanged; stays as `run_network_job.py --timeout` |
| `enforce_deps`, `dep_wait_seconds`, `critical` | unchanged; argv flags |
| `log_name` | `StandardOutPath` + `StandardErrorPath` (same file) |

Key invariant: the renderer already nests `run_with_status →
run_network_job → [env KV…] → python …` as argv. We keep that
verbatim; only the outer "cron line + crontab install" is
replaced.

## 3. File layout

**One plist per job**, at
`~/Library/LaunchAgents/com.stockprediction.<job_id>.plist`.
launchd has no container plist — `StartCalendarInterval` is per
agent. 1:1 with CronJob, allows per-job `launchctl
kickstart`/`disable`, and a shared label prefix makes bulk ops
(`launchctl list | grep com.stockprediction`) trivial.

**LaunchAgents, not LaunchDaemons.** Daemons run as root pre-
login; we need user miniconda env, keychain, ShadowsocksX
proxy. All current jobs assume the user session is live (auto-
login Mac mini).

## 4. Mapping details

- `"20 9 * * 1-5"` → 5 dicts `{Hour=9, Minute=20, Weekday=N}`
  for N=1..5. cron Weekday 1-5 = launchd 1-5. cron `0-4`
  (Sun-Thu, `evening_outlook`) = launchd `{0..4}`. cron
  Sunday is 0 OR 7; launchd Sunday is 0 only — always emit 0.
- `"35 9-15 * * 1-5"` (`risk_check`) → 7 hours × 5 weekdays =
  35 dicts. No job uses `*/N` today; renderer should still
  expand it for future-proofing.
- `env_vars` tuple → top-level `EnvironmentVariables`. Also
  set `PATH=/usr/local/bin:/usr/bin:/bin` and
  `HOME=/Users/wangzilu` — launchd starts near-empty.
- `timeout_sec` stays in `run_network_job.py --timeout`.
  `ExitTimeOut` only governs shutdown, not runtime.
- log redirect → `StandardOutPath` = `StandardErrorPath` =
  `logs/<log_name>` (appends, mirrors `2>&1`). Add
  `WorkingDirectory =
  /Users/wangzilu/MyProjects/stockPrediction`. `RunAtLoad =
  false`, `KeepAlive = false`, `AbandonProcessGroup = false`
  (wrapper timeout kills full tree).

## 5. `scripts/install_launchd.py` (proposed)

Mirrors `install_crontab.py`. Reuses `managed_jobs()` so the
job list stays single-sourced. Subcommands:

- `--dry-run` — render every plist to stdout, no side effects.
- `--apply` — write each plist atomically to
  `~/Library/LaunchAgents/`. For each: `launchctl bootout
  gui/$(id -u) <label>` (ignore "not loaded"), then
  `launchctl bootstrap gui/$(id -u) <path>`. Idempotent —
  byte-diffs plists, skips unchanged.
- `--shadow` — install with `StartCalendarInterval` stripped:
  jobs exist for manual `kickstart` testing while crontab
  still owns prod.
- `--uninstall` — bootout every `com.stockprediction.*` label
  and delete the plist files.

Pre-flight: refuse `--apply` if `logs/` missing or the
miniconda python is unreadable — launchd silently puts the job
in "could not exec" state with no log otherwise.

## 6. Pitfalls

1. **First-load TCC dialog.** `bootstrap` for a new label that
   touches `~/Library`/network may prompt once per label.
   Mitigation: bootstrap one label manually first, then
   automate.
2. **PATH starvation.** Shell-outs (`git`, `caffeinate`) won't
   find binaries unless PATH is set in `EnvironmentVariables`.
3. **Throttle minimum.** launchd refuses to relaunch a job
   that exited <10s ago — irrelevant for our cadence; affects
   `--shadow` kickstart loops only.
4. **Calendar dicts are OR'd.** Always emit full
   `{Hour, Minute, Weekday}` triplets — partials mean "every".
5. **Sleep/wake catch-up.** launchd fires missed
   `StartCalendarInterval` events on wake. For us desirable:
   `enforce_deps` blocks stale runs.

## 7. Rollout

Week 1 — parallel run. Install plists with `--shadow` (no
schedule). Manually `kickstart` 3 representative jobs
(`morning_recommendation`, `qlib_data_update`,
`weekly_full_retrain`) and diff their `logs/*.log` against
crontab-driven runs.

Week 2 — activate launchd schedules **and disable the crontab
block in the same window**. `install_crontab.py` learns a
`--disable` flag that comments out the managed block (not
delete, for rollback). Monitor one full business week +
Saturday cycle.

**Biggest risk: dual-fire.** If both schedulers run any single
job in the same window, `scheduler.job_status` state-file
contention corrupts today's gating, which silently
cross-pollutes downstream `enforce_deps` decisions. The
`--disable` crontab step is therefore a hard precondition for
loading any plist with a `StartCalendarInterval`. Verify with
`crontab -l | grep STOCK_PREDICTION_CX` returning the BEGIN
marker commented out.

Week 3 — delete crontab block; keep `install_crontab.py` in
repo one release as fallback.

## 8. Effort

- Render + plist writer + idempotent `bootstrap` loop: 3h
- Schedule expansion (`9-15` range, `*/N` future-proof): 1h
- `--shadow` / `--uninstall` / pre-flight: 1h
- Parallel-run dry-run harness + log-diff script: 2h
- Manual TCC walkthrough + 1 business cycle observation: 1h

**Total: 6-8h engineering + 1 business week soak.** Backout:
`install_launchd.py --uninstall && install_crontab.py --apply`.
