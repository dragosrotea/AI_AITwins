# DriveMate – CV Engineer Brief

Nov 20, 2026 · @Vlad

Role brief for the computer-vision engineer (camera pipeline, dataset-trained models, visual detectors). Give this brief and the [requirements specification](https://claude.ai/artifact/Wzw87vPMBZMiQp35rTtv4F) to your AI assistant.

## 1. How to use this brief with your AI assistant

Start every new AI session by attaching this brief and the requirements doc, then pasting the prompt below. Work one task ID at a time and commit after each.

```text
You are my pair-programmer for DriveMate, a driver-safety copilot we are building
at the CodeLap 2026 hackathon (about 40 working hours, team of 4).
Attached: (1) the DriveMate requirements specification, (2) my role brief: CV ENGINEER.

Rules:
- I own backend/cv/, training/ (eye_state, yawn, expression), models/ and
  data/download_datasets.sh. Do not edit other folders; if another module needs
  a change, write me a short message I can send its owner.
- The shared contract in section 3 of my brief is the law. Publish exactly
  the topics and fields listed there. If a field seems wrong or missing, tell
  me; do not change backend/schemas.py yourself (the lead owns it).
- Work task by task in the order of section 5. For each task: a short plan,
  then the code, then how to test it, then a commit message like
  "feat(cv): microsleep detector [DD-02] (CV-T09)".
- Real time matters: at least 15 FPS on a laptop CPU. Measure, don't guess.
- Models are trained on public datasets, never calibrated per driver.
  Split train/test BY PERSON, and report precision and recall.
- Every threshold goes in config/thresholds.yaml, never hard-coded.
- The process must also run alone with --fake (synthetic driver.state).
- Never store camera frames to disk outside training data. Never commit
  datasets or model files over 50 MB.
- Before each checkpoint, walk me through the checkpoint checklist in section 4.
Start with task CV-T01.
```

**Tips:** keep one AI chat per checkpoint so the context stays small; paste error messages and printed metrics whole; before any training run, ask the assistant to first inspect the dataset folder and print its structure, because file layouts vary.

## 2. Your scope

You own everything the camera sees: reading the USB webcam, turning the face into numbers with dataset-trained models, and raising the visual alerts (distraction, microsleep, drowsiness, phone in hand).

**You own:** `backend/cv/`, `training/eye_state/`, `training/yawn/`, `training/expression/`, `training/evaluate.py`, `models/`, `data/download_datasets.sh`, the `cv:` section of `config/thresholds.yaml`.

**Requirement IDs:** CV-01–07, DD-01–03, RR-01, ZO-01 (visual part), PH-01–02, ML-01–05 (for camera models), NF-04, NF-09, NF-10, NF-14, NF-15.

**You publish:** `driver.state` (10 Hz), `camera.frame` (5 Hz, small preview), `detector.event` (visual events: distraction, microsleep, drowsiness, phone, no\_face).

**You consume:** nothing required. Optionally `control.trip` to reset rolling windows at trip start.

**Not yours:** deciding what to say or when (the lead's Alert Manager does that), road-rage and zoning-out fusion (lead), anything audio. You deliver clean signals and visual events; the lead combines them.

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

Do the tasks in order; each one ends in a commit. Start the dataset downloads first thing tonight, because some need a sign-up and they take time.

### Before CP0 (now to Fri 23:30): camera running, data downloading

- [ ] **CV-T01 Dataset downloads.** Start now: MRL Eye Dataset, YawDD (free IEEE DataPort account), FER2013 (Kaggle), and send the DMD access request. Write `data/download_datasets.sh` with the steps (manual sign-ups as comments) and `training/DATASETS.md` with each source, licence and size. Datasets stay out of git.
- [ ] **CV-T02 Camera capture.** `backend/cv/camera.py`: open the USB webcam by index from config (add a helper that lists cameras), capture 1280×720, process at 640×360, keep only the latest frame in a thread so processing never lags behind.
- [ ] **CV-T03 Face landmarks.** MediaPipe Face Landmarker (tasks API, video mode, with blendshapes and the facial transformation matrix). The model file downloads through the data script. *Done when:* landmarks track your face live at 15 FPS or more.
- [ ] **CV-T04 Publisher and fake mode.** `backend/cv/run.py` publishes `driver.state` at 10 Hz through `ThreadPublisher` (real `face_present` and `fps`, placeholders for the rest). `--fake` plays scripted sequences (normal, eyes closing, looking down, angry, staring) chosen by a flag, so the lead and UI can test without a camera.
- [ ] **CP0 gate:** your topics appear in the dashboard log; tag `cp0`.

### Before CP1 (Sat 11:00): real signals and the eye model

- [ ] **CV-T05 Geometry features.** Head yaw, pitch and roll in degrees from the transformation matrix (check signs: looking down must give negative pitch, as in the contract); gaze direction from the iris position between the eye corners; eye and mouth aspect ratios kept as fallback features; blink detection.
- [ ] **CV-T06 Eye crops.** One function that cuts both eyes from the landmarks, converts to grayscale and resizes to the model input. Training and live code must use the same function.
- [ ] **CV-T07 Eye-state model v1.** First print the MRL folder structure and a few file names to find how subject and eye state are encoded. Split by subject (about 80/20), train a small CNN (three conv blocks, or MobileNetV3-small) on 64×64 crops, export to ONNX and run it with onnxruntime. *Done when:* `eye_closed_p` is live, the model is under 5 MB, inference is under 3 ms per eye, and precision and recall on held-out subjects are printed.
- [ ] **CV-T08 Camera preview.** Publish `camera.frame` at 5 Hz, 320 px wide JPEG, landmark overlay on or off in config. Never write frames to disk.
- [ ] **CP1 gate:** live webcam drives the dashboard gauges; tag `cp1`.

### Before CP2 (Sat 15:00): visual alerts

- [ ] **CV-T09 Rolling windows.** `eyes_closed_ms` (continuous closure), PERCLOS over 60 s, blink rate per minute, gaze dispersion over 30 s, yawn count over 10 min (filled in CV-T13).
- [ ] **CV-T10 Visual detectors.** Publish `detector.event` for: distraction (DD-01: yaw beyond 30° or pitch below −20° for over 2 s, T2), microsleep (DD-02: closed 1.0 s or more, T3, checked on every frame so it fires within 0.5 s), drowsiness (DD-03: PERCLOS 0.15 or more, or 3 yawns in 10 min, T2), no\_face (CV-06). One event per episode, with a cooldown, and the signal values in `signals`.
- [ ] **CV-T11 Domain check.** Record 2 min of each teammate (normal, blinks, eyes closed, looking down) for local validation only; label with a small key-press tool; measure the eye model on it; set one shared threshold in config (ML-05). Delete the recordings afterwards; never commit them.
- [ ] **CV-T12 Metrics script.** `training/evaluate.py` prints precision, recall and F1 for every model on its held-out people, as a markdown table for the README and deck (ML-02).
- [ ] **CP2 gate:** F0 and F6 acceptance criteria pass; tag `cp2`.

### Before CP3 (Sat 22:00): yawns, anger, zoning out

- [ ] **CV-T13 Yawn classifier.** Run MediaPipe over YawDD videos, extract mouth blendshapes and the mouth ratio per frame, cut 2 s windows, label them from the video's condition (silent, talking or singing, yawning) and the mouth-opening peak, since the videos have no per-frame labels. Train LightGBM, split by person. Publish `yawn_p` and feed `yawns_10m`. *Done when:* talking does not count as yawning on held-out people.
- [ ] **CV-T14 Anger.** Use a pretrained facial-expression model trained on FER2013 (check its licence), run it on the face crop at 5 Hz, smooth `anger_p`, and report anger precision and recall on the FER2013 test split. Fallback if no model works: a blendshape rule (brow down, lips pressed, nose wrinkle).
- [ ] **CV-T15 Visual zoning-out score.** `zoning_visual_p` 0–1 from low gaze dispersion over 60 s, a low blink rate against a population norm in config (not a per-driver baseline), and little head movement. Weights in config. The lead fuses it with vehicle data.
- [ ] **CV-T16 Performance.** Profile and keep 15 FPS or more with every model on: eyes every frame, yawn at 10 Hz, anger at 5 Hz.
- [ ] **CP3 gate:** tag `v0.9-must`.

### Before CP4 (Sun 11:00): extras

- [ ] **CV-T17 Phone in hand.** YOLOv8n on its own thread at 3–5 Hz; set `phone_in_hand`; raise a `phone` event when a phone is visible with gaze down for 1.5 s or more (PH-01, PH-02).
- [ ] **CV-T18 DMD evaluation.** If access arrived, run the distraction and drowsiness detectors on a subset of DMD face-camera clips and add the results to the metrics table.
- [ ] **CV-T19 Demo-spot check.** Test at the real demo spot with glasses on and off; add a desk lamp if needed; freeze thresholds.
- [ ] **CP4 gate:** tag `v1.0`; give the metrics table and dataset notes to the frontend engineer for the README and deck.

### Before CP5 (Sun 12:30)

- [ ] **CV-T20 Rehearsal support.** Stand by during both rehearsals; keep a known-good model set and config on the backup laptop.

## 6. Definition of done for CV

Your part is done when any teammate can sit in front of the webcam with no setup and every visual detection shows up live and correctly.

- [ ] `driver.state` at 10 Hz and 15 FPS or more with all models running on the demo laptop.
- [ ] Distraction, microsleep, drowsiness, no-face and phone events meet DD-01–03, CV-06 and PH-01–02.
- [ ] No calibration step; works for all four teammates, with and without glasses.
- [ ] Metrics table (precision, recall, F1 on held-out people) for the eye, yawn and anger models, delivered for the README and deck.
- [ ] Training scripts reproduce each model; datasets and recordings are not in git; `training/DATASETS.md` lists sources and licences.
- [ ] `--fake` mode still works for the others.
