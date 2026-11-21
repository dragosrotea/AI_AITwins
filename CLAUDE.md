# DriveMate: Claude Code instructions (lead engineer)

DriveMate is a voice-first driver-safety copilot built at the CodeLap 2026 hackathon
(Fri 20 Nov evening to Sun 22 Nov, team of 4). A USB webcam watches the driver,
dataset-trained models detect distraction, drowsiness, zoning out and road rage,
an LLM writes short spoken alerts, and the driver must answer every alert by voice.
A live web dashboard shows everything. Jury requirements: R1 computer-vision detection,
R2 LLM in-car reaction, R3 live demo on a real camera feed, R4 live dashboard.

## Who you are working with

- I am the **lead engineer**. You work only with me.
- My three teammates (CV, Voice & LLM, Frontend & Integrations) write their code by hand
  or with other tools, following the same documents.
- Only change code that belongs to the task I give you. If you need a change in a
  teammate's code, don't make it: write me a short, copy-pasteable message for them
  (file, problem, exact fix).
- Their code may drift from the shared contract. When I merge their work, check it against
  the contract (message schemas and topic list) and tell me what breaks before I merge.

## Key documents (read the part you need, when you need it)

- `docs/requirements.md`: full spec, including the repo layout (section 7). Requirement IDs
  such as `AL-02`, `DD-02`, `RR-05` come from here; look up the exact threshold or
  acceptance criterion before implementing.
- @docs/briefs/lead.md: my task list (`LEAD-T01` ... `LEAD-T26`), the shared contract,
  checkpoints and definition of done.
- `docs/briefs/cv.md`, `docs/briefs/voice.md`, `docs/briefs/ui.md`: what each teammate
  delivers and when. Read them when integrating their parts.

## Commands

```bash
pip install -r requirements.txt
scripts/run_all.sh                      # broker + all processes (missing modules are skipped)
scripts/run_all.sh --fake               # same, every process on synthetic data
scripts/run_all.sh --fake=cv,hr         # only these on synthetic data
python -m backend.bus.broker            # broker alone; GET http://localhost:8765/stats
python -m backend.core.run --fake       # core: vehicle.motion 10 Hz + alert cycle
python -m backend.cv.run --fake         # CV;  python -m backend.voice.run --fake  # voice
python -m backend.integrations.hr.run --sim   # HR (UI owner, not written yet)
python scripts/smoke_test.py --ignore bio.hr  # rates + schema check; --ignore a switched-off topic
python scripts/smoke_test.py            # checks every topic arrives at its expected rate
pytest -q                               # tests
ruff check . && ruff format .           # lint and format
cd dashboard && npm run dev             # dashboard
DRIVEMATE_PROFILE=demo ...              # shortened time windows for the 5-minute demo
```

## Contract (summary; full version in section 3 of the briefs)

- Separate processes talk only through a WebSocket broker at `ws://localhost:8765/bus`.
  Client messages: `{"op":"sub","topics":[...]}` (trailing `*` = prefix match) and
  `{"op":"pub", ...envelope}`.
- Envelope: `{"topic": str, "ts": unix_seconds_float, "src": "cv|voice|core|ui|hr|maps", "data": {...}}`.
- The pydantic schemas file is the single source of truth, and I own it. Adding an optional
  field is fine; renaming or removing one needs the whole team's OK. After any change,
  draft the team announcement for me.
- The Python client offers `BusClient(src)` (async, auto-reconnect) and `ThreadPublisher(src)`
  for camera and audio threads.

| Topic | Producer | Rate |
| --- | --- | --- |
| `driver.state`, `camera.frame` | CV | 10 Hz, 5 Hz |
| `detector.event` | CV (visual events) and core (fusion events) | on event |
| `audio.features`, `voice.status`, `voice.reply` | Voice | 5 Hz, on change, on event |
| `voice.say`, `alert.state`, `control.music`, `checkin.schedule`, `trip.summary`, `vehicle.motion` | Core (me) | on event; `vehicle.motion` 10 Hz |
| `music.volume`, `bio.hr`, `context.road`, `control.trip`, `control.replay`, `control.inject` | UI / integrations | on change; `bio.hr`, `context.road` 1 Hz |

- The core process calls the teammate-written LLM functions `generate_alert`,
  `generate_engagement`, `grade_answer`, `classify_reply` and `summarize_trip`. Each returns
  `None` on timeout or error and never raises. Always handle `None` with a template.

## Rules

- Every threshold, weight, cooldown and time window lives in the config file, never in
  code. The `demo` profile shortens windows.
- Every process must run alone with `--fake`, publishing realistic synthetic data.
- The safety path never waits on the network: LLM timeout 2 s for alerts, then a template.
- Never store raw video or audio. Never commit datasets, model files over 50 MB, `.env` or logs.
- Python 3.11, asyncio, type hints, ruff. Simple working code over clever code: we have hours.
- Logic with rules (Alert Manager state machine, escalation, cooldowns, fusion, check-in
  scheduler, crash detection) gets pytest tests with a fake clock, written with the code.
- One task per session. Start by restating the task ID, its requirement IDs and the acceptance
  criterion, then give a short plan, then the code, then how I verify it.
- End each task with a commit message: `feat(core): alert queue with tiers [AL-01] (LEAD-T10)`.
- For anything touching the contract or a teammate's code, stop and ask me first.

## Checkpoints (merge everything into `main`, run the smoke test, then tag)

| CP | When | Gate |
| --- | --- | --- |
| CP0 | Fri 23:30 | Skeleton, broker, schemas; every process runs with `--fake` (`cp0`) |
| CP1 | Sat 11:00 | Live webcam to dashboard; one alert spoken and its reply classified; replay chart (`cp1`) |
| CP2 | Sat 15:00 | F0, F4, F5, F6 pass; LLM alerts with template fallback; mentor review (`cp2`) |
| CP3 | Sat 22:00 | MUST freeze: F1, F2, F3 pass; demo steps 1-6 run twice (`v0.9-must`) |
| CP4 | Sun 11:00 | Code freeze; extras merged or switched off (`v1.0`) |
| CP5 | Sun 12:30 | Two rehearsals, one offline (`v1.0-final`) |

Merge order at a checkpoint: bus and schemas, core, CV, voice, UI. If a part fails, switch it
back to `--fake` or off in config so `main` always runs.

## Current status (I update this as we go)

- Checkpoint: before CP0
- Done: (none yet)
- Next: LEAD-T01 repo skeleton
- Known issues / decisions: Suunto 9 Peak Pro probably cannot broadcast HR (test SW-00 tonight;
  chest-strap fallback). Vehicle data is a UAH-DriveSet replay, labelled as replay.
