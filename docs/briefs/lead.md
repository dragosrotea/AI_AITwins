# DriveMate – Lead Engineer Brief

Nov 20, 2026 · @Vlad

Role brief for the lead engineer (integration, Alert Manager, fusion, vehicle data, crash flow). Give this brief and the [requirements specification](https://claude.ai/artifact/Wzw87vPMBZMiQp35rTtv4F) to your AI assistant.

## 1. How to use this brief with your AI assistant

Start every new AI session by attaching this brief and the requirements doc, then pasting the prompt below. Work one task ID at a time and commit after each.

```text
You are my pair-programmer for DriveMate, a driver-safety copilot we are building
at the CodeLap 2026 hackathon (about 40 working hours, team of 4).
Attached: (1) the DriveMate requirements specification, (2) my role brief: LEAD ENGINEER.

Rules:
- I own the bus, schemas, core process (Alert Manager, fusion, scheduler,
  vehicle replay, crash flow) and integration. Other people own cv/, voice/,
  audio/, llm/, dashboard/ and integrations/. Do not edit their code; if their
  module needs a change, write me a short message I can send them.
- The shared contract in section 3 of my brief is the law. I am its owner:
  if we change it, update backend/schemas.py AND tell me exactly what to
  announce to the team.
- Work task by task in the order of section 5. For each task: a short plan,
  then the code, then how to test it, then a commit message like
  "feat(core): alert queue with tiers [AL-01] (LEAD-T07)".
- Simple, working code beats clever code. Python 3.11, asyncio, type hints,
  pytest for logic with rules (scheduler, fusion, escalation).
- Every threshold goes in config/*.yaml, never hard-coded. Secrets in .env.
- Every process must also run alone with --fake.
- Never store raw video or audio. Never commit datasets, big model files or .env.
- Before each checkpoint, walk me through the checkpoint checklist in section 4.
Start with task LEAD-T01.
```

**Tips:** keep one AI chat per checkpoint so the context stays small; paste error messages whole; when the assistant suggests changing the contract, stop and decide yourself, because three other people depend on it.

## 2. Your scope

You own everything that connects the other three parts and decides what the driver hears: the message bus, the shared schemas, the core process and every integration checkpoint.

**You own:** `backend/bus/`, `backend/schemas.py`, `backend/core/` (Alert Manager, fusion, check-in scheduler, vehicle replay, crash flow), `config/thresholds.yaml` structure (plus its `core:` section), `scripts/`, `docs/`.

**Requirement IDs:** AL-01–05, CI-02–05 (logic), DD-04, RR-04–07 (fusion and intervention flow), ZO-01 (fusion with vehicle context), ZO-02, ZO-05, ZO-06, CR-01–03, MAP-03 (holding prompts), PT-01 (trigger), ML-03, NF-01, NF-05, NF-06, NF-13, the demo script and every checkpoint.

**You publish:** `vehicle.motion`, `voice.say`, `alert.state`, `detector.event` (fusion events: road\_rage, zoning\_out, crash\_candidate, checkin\_due, break\_due), `control.music`, `checkin.schedule`, `trip.summary`.

**You consume:** `driver.state`, `detector.event` (visual events from CV), `audio.features`, `voice.status`, `voice.reply`, `bio.hr`, `context.road`, `music.volume`, `control.trip`, `control.replay`, `control.inject`.

**Not yours:** camera models, speech in or out, prompt wording, dashboard code. You call the `backend/llm` functions the voice engineer writes; you do not write prompts.

### Folder ownership (whole team)

| Folder or file | Owner |
| --- | --- |
| `backend/bus/`, `backend/schemas.py`, `backend/core/`, `scripts/`, `docs/` | Lead |
| `backend/cv/`, `training/`, `models/`, `data/download_datasets.sh` | CV |
| `backend/voice/`, `backend/audio/`, `backend/llm/`, `config/prompts/`, `config/lexicons/`, `config/trivia_fallback.json`, `assets/audio/` | Voice & LLM |
| `dashboard/`, `backend/integrations/`, `assets/music/`, `README.md`, `presentation/` | Frontend & Integrations |
| `config/thresholds.yaml` | Shared: each owner edits only their own section (`core:`, `cv:`, `voice:`, `ui:`) |

## 3. Shared contract (identical in all four briefs)

Every part of DriveMate is a separate process that talks only through a WebSocket message broker, so each person can build and test alone and one crash never stops the rest. The lead owns this contract; nobody else changes it.

### Broker and client

- **Broker:** `ws://localhost:8765/bus`, started with `python -m backend.bus.broker`.
- **Send to broker:** `{"op": "sub", "topics": ["driver.state", "alert.*"]}` to subscribe (a trailing `*` matches a prefix); `{"op": "pub", ...envelope}` to publish.
- **Python client:** `backend/bus/client.py` gives `BusClient(src)` with `await connect()`, `await publish(topic, data)`, `subscribe(topics, handler)`, and `ThreadPublisher(src)` for code running in a normal thread (camera loop, audio callback).
- **Browser:** the dashboard opens the same WebSocket and sends the same JSON.

### Envelope (every message)

```json
{"topic": "driver.state", "ts": 1763660000.12, "src": "cv", "data": { }}
```

`ts` is Unix time in seconds (float). `src` is one of `cv`, `voice`, `core`, `ui`, `hr`, `maps`. Field schemas live as pydantic v2 models in `backend/schemas.py`; the dashboard mirrors them in `dashboard/src/types.ts`.

### Topics

| Topic | Owner | Rate | `data` fields |
| --- | --- | --- | --- |
| `driver.state` | CV | 10 Hz | `face_present`, `eye_closed_p`, `eyes_closed_ms`, `perclos_60s`, `blink_rate_pm`, `gaze_dispersion_30s`, `head` {`yaw`, `pitch`, `roll`} in degrees, `yawn_p`, `yawns_10m`, `anger_p`, `zoning_visual_p`, `phone_in_hand`, `fps` |
| `camera.frame` | CV | 5 Hz | `jpeg_b64` (320 px wide, optional landmark overlay); never written to disk |
| `detector.event` | CV (visual), Lead (fusion) | on event | `event_id`, `type` (distraction, microsleep, drowsiness, phone, no\_face, road\_rage, zoning\_out, crash\_candidate, checkin\_due, break\_due), `tier` (T1/T2/T3), `score` 0–100, `signals` {} |
| `audio.features` | Voice | 5 Hz | `loudness_db`, `baseline_db`, `shouting`, `speaking`, `swear_count_60s`, `anger_voice_p` (null until the stretch model exists) |
| `voice.say` | Lead | on event | `alert_id`, `text`, `tier`, `lang` (ro/en), `listen_s` (default 8), `audio_key` (optional pre-rendered file) |
| `voice.status` | Voice | on change | `alert_id`, `state` (speaking, listening, idle) |
| `voice.reply` | Voice | on event | `alert_id`, `transcript`, `intent` (OK, NOT\_OK, ANSWER, HELP, UNCLEAR, NONE), `latency_ms`, `path` (keyword/llm) |
| `alert.state` | Lead | on change | `alert_id`, `type`, `tier`, `status` (queued, held, speaking, listening, closed, escalated), `text`, `reply`, `intent`, `latency_ms` |
| `vehicle.motion` | Lead | 10 Hz | `accel_long_g`, `accel_lat_g`, `accel_mag_g`, `speed_kmh`, `lat`, `lon`, `road` (motorway/secondary), `source` ("replay") |
| `control.music` | Lead | on event | `duck` (bool), `level` 0–1 |
| `music.volume` | UI | on change | `volume` 0–100, `playing` |
| `bio.hr` | UI (integrations) | 1 Hz | `bpm`, `rr_ms` \[\], `source` (suunto, strap, sim) |
| `context.road` | UI (integrations) | 1 Hz | `load` (LOW/MEDIUM/HIGH), `traffic`, `next_maneuver`, `dist_m` |
| `checkin.schedule` | Lead | on change | `interval_min`, `next_checkin_ts`, `next_break_ts` |
| `control.trip` | UI | on event | `action` (start/stop), `pretrip` {`planned_min`, `sleep_h`, `awake_h`, `tiredness`, `caffeine`, `engagement`, `lang`, `emergency_contact`, `resting_hr`} |
| `control.replay` | UI | on event | `action` (play, pause, seek), `trip`, `segment` |
| `control.inject` | UI | on event | `event` (harsh\_brake, crash) |
| `trip.summary` | Lead | trip end | `text`, `stats` {} |

### LLM functions (written by Voice & LLM in `backend/llm/`, called by the lead's core process)

```python
async def generate_alert(ctx: dict, timeout_s: float = 2.0) -> str | None
async def generate_engagement(ctx: dict, timeout_s: float = 2.0) -> dict | None   # {kind: "trivia"|"fact", text, answer}
async def grade_answer(question: str, expected: str, reply: str, timeout_s: float = 1.5) -> dict | None  # {correct, feedback}
async def classify_reply(transcript: str, expecting: str, timeout_s: float = 1.5) -> str | None  # intent
async def summarize_trip(log: list[dict], timeout_s: float = 10.0) -> str | None
```

Every function returns `None` on timeout or error; the caller then uses a template. No function ever raises into the caller.

### Conventions

- Python 3.11, type hints, `ruff` for formatting, `pytest` for logic. Node 20 for the dashboard.
- Every process accepts `--fake` (or `--sim`) and then publishes realistic synthetic data for its own topics, so nobody waits for anybody.
- Thresholds in `config/thresholds.yaml` (one section per owner); secrets in `.env`, with `.env.example` committed.
- Logs go to `logs/` as JSONL. Never store raw video or audio.
- Run everything: `scripts/run_all.sh` (broker, core, cv, voice, hr); dashboard: `cd dashboard && npm run dev`.
- Commit message format: `feat(cv): microsleep detector [DD-02] (CV-T09)`; branch names `lead/*`, `cv/*`, `voice/*`, `ui/*`.

## 4. Checkpoints and integration protocol (identical in all four briefs)

We merge everyone's work into `main` at six fixed checkpoints, and `main` must run end to end after each one. Between checkpoints, merge small pieces whenever the smoke test passes; never save integration for the end.

| CP | When | Name | Integration test that must pass on the demo laptop | Tag |
| --- | --- | --- | --- | --- |
| CP0 | Fri 23:30 | Skeleton and contracts | Broker runs; `backend/schemas.py` merged; every process starts with `--fake` and its topics show in the dashboard message log; `scripts/smoke_test.py` green | `cp0` |
| CP1 | Sat 11:00 | First live loop | Live webcam drives `driver.state` on the dashboard; a hard-coded distraction alert is spoken and the reply is classified; the vehicle replay draws the acceleration chart | `cp1` |
| CP2 | Sat 15:00 | Core loop (mentor review) | F0, F4, F5, F6 acceptance criteria pass; eye-state model v1 live; alerts written by the LLM with template fallback; demo shown to mentors | `cp2` |
| CP3 | Sat 22:00 | MUST freeze | F1, F2, F3 pass; demo script steps 1–6 run end to end twice in a row without a restart | `v0.9-must` |
| CP4 | Sun 11:00 | Code freeze | Extras that pass their tests are merged, the rest switched off in config; README draft; model metrics table | `v1.0` |
| CP5 | Sun 12:30 | Rehearsal ready | Two full rehearsals, one with Wi-Fi off; deck in the repo; only bug fixes after this | `v1.0-final` |

### Before every checkpoint (each person)

- [ ] T−30 min: stop starting new work; finish or switch off what is half-done (config flag or `--fake`).
- [ ] Rebase your branch on `main`, run `pytest` and `scripts/smoke_test.py` locally.
- [ ] Run your process against the real broker together with the others' `--fake` processes.
- [ ] Open a pull request listing the task IDs done and anything that moved to the next checkpoint.
- [ ] Update your section of `config/thresholds.yaml` and the README notes for your part.

### At every checkpoint (lead runs it, everyone present, 20 min)

1. Merge in order: bus and schemas, then core, CV, voice, UI.
2. Run `scripts/run_all.sh` and the dashboard on the demo laptop, then `scripts/smoke_test.py`, which checks every topic arrives at its expected rate.
3. Run the checkpoint's integration test from the table above. If it passes, tag it and push.
4. If one part fails, it goes back behind its `--fake` or config flag so `main` still runs; its owner fixes it on a branch.
5. 5-minute re-plan: what passed, what slipped, who helps whom before the next checkpoint.

### Rules between checkpoints

- Contract changes (new topic or field) go through the lead, who updates `backend/schemas.py` and tells everyone in the team chat. Adding an optional field is fine; renaming or removing one needs everyone's OK.
- If you are blocked for more than 20 minutes, say so in the team chat with what you tried.
- Pull often: `git pull --rebase` at least every 2 hours.

## 5. Tasks by checkpoint

Do the tasks in order; each one ends in a commit. Your CP0 tasks unblock everyone else, so they come before anything of your own.

### Before CP0 (now to Fri 23:30): skeleton and contracts

- [ ] **LEAD-T01 Repo skeleton.** Create the folders from section 2 and the requirements doc (section 7), `.gitignore` (`data/`, large files in `models/`, `.env`, `logs/`, `node_modules/`), `.env.example`, `requirements.txt`, `pyproject.toml` with ruff. *Done when:* a fresh clone installs with `pip install -r requirements.txt`.
- [ ] **LEAD-T02 Broker.** `backend/bus/broker.py` on port 8765, path `/bus`: tracks each connection's subscriptions, fans out with prefix wildcards, gives every client a bounded queue (about 200 messages, oldest dropped) so a slow dashboard never blocks the camera, and serves per-topic message rates at `GET /stats`. *Done when:* two test clients exchange 1,000 messages per second without growing memory.
- [ ] **LEAD-T03 Client library.** `backend/bus/client.py`: async `BusClient` with auto-reconnect every 1 s, and `ThreadPublisher` (thread-safe queue plus its own asyncio loop) for camera and audio threads. *Done when:* pytest covers publish, subscribe, wildcard and reconnect.
- [ ] **LEAD-T04 Schemas.** `backend/schemas.py`: one pydantic v2 model per topic in section 3, a `TOPIC_MODELS` map and a `validate(topic, data)` helper; `scripts/export_schema.py` writes JSON Schema to `docs/contract.json` for the dashboard types. *Done when:* the other three confirm the fields match their brief.
- [ ] **LEAD-T05 Config.** `config/thresholds.yaml` with `core`, `cv`, `voice`, `ui` sections plus a `demo` profile that shortens time windows; `backend/config.py` loads it, with the profile chosen by `DRIVEMATE_PROFILE=demo`.
- [ ] **LEAD-T06 Run and smoke scripts.** `scripts/run_all.sh` starts broker, core, cv, voice and hr (each with an optional `--fake`); `scripts/smoke_test.py` listens for 10 s and prints each topic's rate against the expected rate in section 3, exiting non-zero if one is missing. Core gets a `--fake` mode publishing `vehicle.motion` and cycling `alert.state`.
- [ ] **LEAD-T07 Vehicle data.** Download UAH-DriveSet, print the first lines of every file in one trip folder, and write in `docs/datasets.md` which files hold acceleration, speed and GPS, with their units and sample rates. Do not assume the layout; check it.
- [ ] **CP0 gate:** run the checkpoint (section 4) and tag `cp0`.

### Before CP1 (Sat 11:00): first live loop

- [ ] **LEAD-T08 Vehicle replay.** `backend/core/vehicle.py`: load three trips (motorway normal, drowsy, aggressive), resample to 10 Hz, publish `vehicle.motion` with `source: "replay"`; obey `control.replay` (play, pause, seek, choose trip). Check acceleration units against T07 before converting to g. *Done when:* the dashboard chart shows real braking and turns, and speed matches the GPS track.
- [ ] **LEAD-T09 Event injector.** On `control.inject`: `harsh_brake` overlays −0.6 g for 1.5 s; `crash` adds a 5 g spike, then ramps speed to 0 within 3 s.
- [ ] **LEAD-T10 Alert Manager v1.** `backend/core/alerts.py`: one state machine per alert (queued, speaking, listening, closed or escalated); a priority queue by tier; one alert active at a time; T3 pre-empts. Publish `voice.say` and every `alert.state` change; duck music via `control.music` at start and restore at the end. Reply handling: OK closes; NOT\_OK calls a follow-up hook; UNCLEAR or no reply within `listen_s` + 3 s repeats once with a shorter template, then escalates a tier (VR-05); HELP calls the emergency hook. *Done when:* pytest with a fake clock covers every path.
- [ ] **LEAD-T11 Hard-coded first alert.** A CV `distraction` event becomes a T2 alert with a fixed template in the driver's language. This is the CP1 integration test.
- [ ] **LEAD-T12 Trip log.** Every event, alert, spoken text, reply, intent and latency goes to `logs/trip_<timestamp>.jsonl` (AL-04).
- [ ] **CP1 gate:** tag `cp1`.

### Before CP2 (Sat 15:00): core loop and mentor review

- [ ] **LEAD-T13 LLM-written alerts.** Call `llm.generate_alert(ctx)` with a 2 s timeout; on `None`, use a template from `config/prompts/templates.yaml` (written by the voice engineer). Log which path was used (AL-05).
- [ ] **LEAD-T14 Rate limits and cooldowns.** At most one T1 per 5 min and one non-critical prompt per 2 min (AL-02); a per-type cooldown table in config.
- [ ] **LEAD-T15 Visual event routing.** Map each CV event type to its tier; a drowsiness or microsleep event cancels any queued trivia (DD-04).
- [ ] **LEAD-T16 Mentor demo.** Run F0, F4, F5, F6 against their acceptance criteria; show the mentors at 15:00 and write their feedback in the team chat.
- [ ] **CP2 gate:** tag `cp2`.

### Before CP3 (Sat 22:00): MUST freeze

- [ ] **LEAD-T17 Road-rage fusion.** `backend/core/fusion.py` (RR-04, RR-05, RR-07): anger held for 3 s from `driver.state`, `swear_count_60s` and `shouting` from `audio.features`, harsh events in the last 60 s from `vehicle.motion`, the heart-rate stress cue (SW-03) and rising music volume. Score 0–100 with weights in config. Trigger at 60 only if 2 of 3 modalities agree, or the face alone stays angry for 5 s; a firmer message at 85; 3-minute cooldown and no more than 3 in 30 min. *Done when:* pytest proves swearing alone never triggers.
- [ ] **LEAD-T18 Calm-down flow.** On `road_rage`: duck music to 0.7, then speak the LLM's calm line (RR-06).
- [ ] **LEAD-T19 Zoning-out fusion and trivia flow.** `zoning_visual_p` from CV plus monotony from `vehicle.motion` (above 70 km/h with under 5 km/h spread for 10 min; 60 s in the demo profile), held when `context.road` load is HIGH. Raise a T1 `zoning_out` event, get `llm.generate_engagement`, speak it; on an ANSWER reply call `llm.grade_answer` and speak the feedback. After 3 engagements in 30 min, recommend a break instead (ZO-02, ZO-05, ZO-06).
- [ ] **LEAD-T20 Check-in scheduler.** On `control.trip` start, compute the interval with the CI-02 rule and publish `checkin.schedule`; adapt live (CI-04); break reminders (CI-03); a NOT\_OK reply offers a break and shortens the interval. *Done when:* pytest shows 5 h sleep and 21:00 departure gives 15 min, with a break at 1:45.
- [ ] **LEAD-T21 Demo dry run.** Tune the `demo` profile and run demo script steps 1–6 twice in a row; fix whatever breaks.
- [ ] **CP3 gate:** tag `v0.9-must`.

### Before CP4 (Sun 11:00): extras and code freeze

- [ ] **LEAD-T22 Crash flow.** Detect a crash candidate (4 g or more, then speed under 5 km/h within 10 s); start a T3 "Are you okay?" with a 20 s countdown (add an optional `countdown_s` field to `alert.state` and announce it). OK cancels; HELP or silence calls and texts the emergency contact with the location through Twilio. Without Twilio keys, show a dashboard banner instead.
- [ ] **LEAD-T23 Hold prompts in busy zones.** T1 prompts wait while road load is HIGH (AL-03).
- [ ] **LEAD-T24 Trip summary.** On trip stop, call `llm.summarize_trip(log)` and publish `trip.summary`; fall back to a stats-only text.
- [ ] **LEAD-T25 Resilience test.** Kill each process in turn during a run and confirm the rest continue and it reconnects (NF-06); do a full run with Wi-Fi off (NF-05).
- [ ] **CP4 gate:** tag `v1.0`.

### Before CP5 (Sun 12:30): rehearsal

- [ ] **LEAD-T26 Rehearsals.** Two timed runs of the full demo script, one offline; a backup laptop with the repo cloned and models downloaded; tag `v1.0-final`.

## 6. Definition of done for the lead

Your part is done when the whole system runs from one command on the demo laptop and survives any one part failing.

- [ ] `scripts/run_all.sh` plus the dashboard start everything; `scripts/smoke_test.py` is green.
- [ ] Every alert path (OK, NOT\_OK, silence, HELP, T3 pre-emption) is covered by pytest.
- [ ] Rage fusion, zoning-out flow, check-in scheduler and crash flow pass their acceptance criteria in the requirements doc.
- [ ] Killing any one process does not stop the others, and it reconnects on restart.
- [ ] Full demo script runs twice in a row, once with Wi-Fi off.
- [ ] Tags `cp0` to `v1.0-final` exist, and `docs/contract.json` matches `backend/schemas.py`.
