# DriveMate – Frontend & Integrations Engineer Brief

Nov 20, 2026 · @Vlad

Role brief for the frontend and integrations engineer (dashboard, music player, heart rate, maps, README and deck). Give this brief and the [requirements specification](https://claude.ai/artifact/Wzw87vPMBZMiQp35rTtv4F) to your AI assistant.

## 1. How to use this brief with your AI assistant

Start every new AI session by attaching this brief and the requirements doc, then pasting the prompt below. Work one task ID at a time and commit after each.

```text
You are my pair-programmer for DriveMate, a driver-safety copilot we are building
at the CodeLap 2026 hackathon (about 40 working hours, team of 4).
Attached: (1) the DriveMate requirements specification, (2) my role brief:
FRONTEND & INTEGRATIONS ENGINEER.

Rules:
- I own dashboard/ (React + Vite), backend/integrations/ (heart rate, maps),
  assets/music/, README.md and presentation/. Do not edit other folders; if
  another module needs a change, write me a short message I can send its owner.
- The shared contract in section 3 of my brief is the law. The dashboard reads
  and writes exactly the topics and fields listed there. Generate TypeScript
  types that mirror backend/schemas.py; do not change the schema yourself
  (the lead owns it).
- Work task by task in the order of section 5. For each task: a short plan,
  then the code, then how to test it, then a commit message like
  "feat(ui): acceleration chart [DB-02] (UI-T05)".
- The Driver view must be readable at a glance (under 2 s): one status colour,
  one big message, nothing to tap while moving.
- Charts update live at 5 Hz or more without freezing the page: keep a
  60-second ring buffer per series, no unbounded arrays.
- API keys in .env (VITE_ prefix only for keys safe in the browser).
- Every integration process must also run alone with --sim.
- Before each checkpoint, walk me through the checkpoint checklist in section 4.
Start with task UI-T01.
```

**Tips:** keep one AI chat per checkpoint so the context stays small; run the dashboard against the --fake processes so you never wait for others; screenshot broken layouts and paste them to the assistant.

## 2. Your scope

You own everything the jury and the driver look at, plus the outside data sources: the live dashboard (Driver view and Co-pilot view), the music player, heart rate, Google Maps, and the README and deck.

**You own:** `dashboard/`, `backend/integrations/hr/`, `backend/integrations/maps/`, `assets/music/`, `README.md`, `presentation/`, the `ui:` section of `config/thresholds.yaml`.

**Requirement IDs:** DB-01–07, CI-01 and CI-06 (pre-trip form), SW-00–05, MAP-01, MAP-02, MAP-04, MAP-05 (replay controls), CR-04 (simulate-crash button), NF-07, and the README and deck parts of the Definition of Done.

**You publish:** `music.volume`, `bio.hr`, `context.road`, `control.trip`, `control.replay`, `control.inject`.

**You consume:** every topic (the Co-pilot view shows all of them), and you obey `control.music` (duck or restore the player volume).

**Not yours:** deciding alerts (lead), what the voice says (voice engineer), camera models (CV). The dashboard shows and controls; it never decides.

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

Do the tasks in order; each one ends in a commit. Build every panel against the `--fake` processes so you never wait for the others.

### Before CP0 (now to Fri 23:30): dashboard shell, Suunto test

- [ ] **UI-T01 Scaffold and bus client.** Vite + React + TypeScript in `dashboard/`; `src/bus.ts` opens the broker WebSocket with auto-reconnect, subscribes to `*`, keeps the latest message per topic and a 60 s ring buffer per chart series (a small store such as zustand).
- [ ] **UI-T02 Types.** `src/types.ts` mirrors `backend/schemas.py`; generate it from `docs/contract.json` once the lead exports it.
- [ ] **UI-T03 Message log.** A debug page listing every topic, its live rate and last payload. This is the CP0 integration test.
- [ ] **UI-T04 Layout.** Two routes: `/driver` and `/copilot`, with empty panels in their final positions.
- [ ] **UI-T05 Suunto test (SW-00).** Start an exercise on the 9 Peak Pro, then run `python -m backend.integrations.hr.scan` (bleak) to list devices offering the standard Heart Rate Service (UUID 0x180D). Post the result in the team chat for the lead's decision log; if it fails, find a Bluetooth chest strap for Saturday.
- [ ] **UI-T06 Music player.** Royalty-free MP3s in `assets/music/`, an HTML audio element with a volume slider; publish `music.volume` on every change; obey `control.music` (duck to `level`, then restore).
- [ ] **CP0 gate:** every `--fake` topic visible in the log; tag `cp0`.

### Before CP1 (Sat 11:00): live charts

- [ ] **UI-T07 Co-pilot charts.** Acceleration (longitudinal and lateral g, last 60 s, markers at 0.4 g, a visible "replay" label); music volume with cabin loudness in dB; driver gauges (eye closure, PERCLOS, head yaw and pitch, anger, yawn); a debug corner with camera FPS and message delay (now minus `ts`).
- [ ] **UI-T08 Camera preview.** Show `camera.frame` images, with the overlay as CV sends it.
- [ ] **CP1 gate:** live webcam and replay visible; tag `cp1`.

### Before CP2 (Sat 15:00): conversation, driver view, form

- [ ] **UI-T09 Conversation panel.** From `alert.state`, `voice.status` and `voice.reply`: current message text, speaking or listening indicator, listen-window countdown, the driver's transcript, intent and latency.
- [ ] **UI-T10 Event timeline.** Every `detector.event` and closed alert, newest first, coloured by tier.
- [ ] **UI-T11 Driver view.** Full-screen status colour (green, amber, red by the highest active tier), the current spoken message in very large text, the next check-in countdown. Nothing to tap (NF-07).
- [ ] **UI-T12 Pre-trip form.** CI-01 fields, including resting heart rate; publishes `control.trip` start; disabled while `vehicle.motion` speed is above 5 km/h (CI-06); a stop-trip button.
- [ ] **CP2 gate:** F5 acceptance criteria pass; tag `cp2`.

### Before CP3 (Sat 22:00): scores, controls, heart rate

- [ ] **UI-T13 Attention score and cue tags.** Attention score 0–100 with its parts (formula and weights in the `ui:` config section); emotional cue tags (DB-04): "stress" when volume rises and anger is above 0.5 within 30 s; "low engagement" when volume is low and the zoning-out score is high.
- [ ] **UI-T14 Replay controls and injector.** Play, pause, choose trip and segment (`control.replay`); "Harsh brake" and "Simulate crash (demo)" buttons (`control.inject`, CR-04).
- [ ] **UI-T15 Live heart rate.** `backend/integrations/hr/run.py` with bleak: connect to the configured device (the Suunto if SW-00 passed, otherwise the strap), subscribe to the Heart Rate Measurement characteristic (0x2A37), parse the flags byte (8- or 16-bit bpm, optional RR intervals in 1/1024 s), publish `bio.hr` at 1 Hz, reconnect automatically; `--sim` for a synthetic signal. An HR panel with the resting line from the form.
- [ ] **UI-T16 Check-in display.** Show `checkin.schedule`: interval, next check-in and next break.
- [ ] **CP3 gate:** tag `v0.9-must`.

### Before CP4 (Sun 11:00): maps, README, deck

- [ ] **UI-T17 Maps (X2).** `backend/integrations/maps/run.py`: call the Routes API once for the replay route (start and end from the replay's GPS, which is in the Madrid area) with traffic; each second, compute road load from the next manoeuvre (under 500 m), traffic level and road type at the current replay position, and publish `context.road`. Use the Places API for fuel stations and rest areas ahead (ask the lead to add an optional `rest_area` field). A map panel with route, position and load colour. Restrict the API key and check billing first.
- [ ] **UI-T18 Trip summary view.** Show `trip.summary` at trip end.
- [ ] **UI-T19 README.** Idea, architecture image (from the requirements doc), install and run steps, dependencies, tools used (including AI assistants), datasets and licences plus the metrics table (from CV), team.
- [ ] **UI-T20 Deck.** Official template, named `AI_<TeamName>.pptx`, in `presentation/`. Slides: problem, solution, live demo flow, architecture, models and datasets with metrics, privacy, impact, next steps.
- [ ] **CP4 gate:** tag `v1.0`.

### Before CP5 (Sun 12:30)

- [ ] **UI-T21 Final polish.** Screenshots in the README; projector check (font sizes, contrast); a timer view for the rehearsal.

## 6. Definition of done for Frontend & Integrations

Your part is done when the jury can follow everything happening in the car from the Co-pilot view, the driver only ever needs a glance, and the repo explains itself.

- [ ] Co-pilot view shows every topic live (DB-01–07) without freezing over a 10-minute run.
- [ ] Driver view readable in under 2 s from a metre away; nothing to tap while moving.
- [ ] Live heart rate from the Suunto or a chest strap, with automatic reconnect; `--sim` fallback.
- [ ] Replay controls and the demo injector work; the replay is clearly labelled.
- [ ] Maps panel and `context.road` work, or are cleanly switched off in config.
- [ ] README covers idea, architecture, install and run, dependencies, tools, datasets, licences and metrics; the deck is in `presentation/` on the official template.
