# DriveMate – Voice & LLM Engineer Brief

Nov 20, 2026 · @Vlad

Role brief for the voice and LLM engineer (speech in and out, reply understanding, audio signals, all LLM prompts). Give this brief and the [requirements specification](https://claude.ai/artifact/Wzw87vPMBZMiQp35rTtv4F) to your AI assistant.

## 1. How to use this brief with your AI assistant

Start every new AI session by attaching this brief and the requirements doc, then pasting the prompt below. Work one task ID at a time and commit after each.

```text
You are my pair-programmer for DriveMate, a driver-safety copilot we are building
at the CodeLap 2026 hackathon (about 40 working hours, team of 4).
Attached: (1) the DriveMate requirements specification, (2) my role brief:
VOICE & LLM ENGINEER.

Rules:
- I own backend/voice/, backend/audio/, backend/llm/, config/prompts/,
  config/lexicons/, config/trivia_fallback.json and assets/audio/. Do not edit
  other folders; if another module needs a change, write me a short message I
  can send its owner.
- The shared contract in section 3 of my brief is the law. Publish and consume
  exactly the topics listed there, and implement the backend/llm functions with
  exactly the signatures listed. Do not change backend/schemas.py yourself
  (the lead owns it); tell me if something is missing.
- Work task by task in the order of section 5. For each task: a short plan,
  then the code, then how to test it, then a commit message like
  "feat(voice): keyword fast-path classifier [VR-04] (VOICE-T06)".
- Latency matters: measure every step (TTS start, listen window, STT, LLM).
- Every LLM call has a timeout and a template fallback. Safety never waits on
  the network.
- Romanian and English must both work.
- Thresholds and word lists go in config/, never in code. API keys in .env.
- The process must also run alone with --fake.
- Never store audio. Keep transcripts only for replies to alerts.
- Before each checkpoint, walk me through the checkpoint checklist in section 4.
Start with task VOICE-T01.
```

**Tips:** keep one AI chat per checkpoint so the context stays small; test with real speech from all four teammates early, not only your own voice; paste latency logs whole when asking for speed-ups.

## 2. Your scope

You own every word the driver hears and every word they say back: text-to-speech, the listening window, speech-to-text, reply understanding, the cabin-audio signals for road rage, and all LLM prompts.

**You own:** `backend/voice/`, `backend/audio/`, `backend/llm/`, `config/prompts/`, `config/lexicons/`, `config/trivia_fallback.json`, `assets/audio/` (pre-rendered alerts and alarm tones), the `voice:` section of `config/thresholds.yaml`.

**Requirement IDs:** VR-01–07, LLM-01–05, RR-02, RR-03, RR-06 (message wording), ZO-03, ZO-04, ZO-07, CI-05 (check-in question wording), PT-01 (summary text), NF-02, NF-03, NF-11; stretch: angry-voice model (ML, CREMA-D).

**You publish:** `audio.features` (5 Hz), `voice.status`, `voice.reply`.

**You consume:** `voice.say` (from the lead's Alert Manager), `control.trip` (driver's language).

**You provide as a library:** the `backend/llm` functions in section 3, which the lead's core process imports and calls.

**Not yours:** deciding when to speak or which alert wins (Alert Manager), music volume (the dashboard player obeys `control.music` from the lead), the camera.

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

Do the tasks in order; each one ends in a commit. Ship stub versions of the LLM functions by CP0 so the lead can wire them in while you build the real ones.

### Before CP0 (now to Fri 23:30): sound in, sound out, stubs

- [ ] **VOICE-T01 Audio devices.** List input and output devices; choose them in config. One microphone capture thread (16 kHz mono, 30 ms frames) fills a ring buffer that both cabin monitoring and the reply window read. Never open the mic twice.
- [ ] **VOICE-T02 Text to speech.** edge-tts with a Romanian and an English voice; play through the chosen output; measure time to first audio. `backend/voice/prerender.py` renders every template and the alarm and chime tones into `assets/audio/` so speech works offline.
- [ ] **VOICE-T03 Speech to text.** Load faster-whisper (base, int8) once at startup; transcribe a 3 s clip in each language; log the latency.
- [ ] **VOICE-T04 LLM client and stubs.** `backend/llm/client.py` reads the provider and key from `.env`, is async, applies the timeout and returns `None` on any error. Stub all five functions from section 3 with canned answers so the lead can call them tonight. Measure one real call's latency.
- [ ] **VOICE-T05 Fake mode.** `backend/voice/run.py --fake`: on `voice.say`, wait 2 s and publish `voice.reply` OK with a random latency; publish synthetic `audio.features`.
- [ ] **CP0 gate:** your topics appear in the dashboard log; tag `cp0`.

### Before CP1 (Sat 11:00): the voice loop

- [ ] **VOICE-T06 Voice loop.** On `voice.say`: play the pre-rendered file if `audio_key` exists, else TTS; publish `voice.status` speaking, then a short chime and listening; capture with voice-activity detection for up to `listen_s`, stopping 0.8 s after speech ends; transcribe; classify; publish `voice.reply` with `latency_ms` from end of speech to intent. While busy, ignore new `voice.say` unless it is T3, which interrupts.
- [ ] **VOICE-T07 Keyword fast path.** `config/lexicons/intents.yaml` in RO and EN. OK: yes, yeah, sure, ok, fine, I'm good, da, sigur, bine, sunt ok. NOT\_OK: tired, not really, no, obosit, nu prea. HELP: help, call someone, ajutor, sună. Ignore case and diacritics; return `None` when nothing matches, so the LLM decides (VR-04).
- [ ] **VOICE-T08 Cabin audio features.** `backend/audio/`: publish `audio.features` at 5 Hz with loudness in dB, a rolling 60 s baseline, `shouting` (15 dB or more above baseline for 1 s or longer), and `speaking` from voice activity. Mute monitoring while our own TTS plays, so DriveMate never hears itself.
- [ ] **CP1 gate:** a spoken alert gets a classified reply on the dashboard; tag `cp1`.

### Before CP2 (Sat 15:00): LLM and accuracy

- [ ] **VOICE-T09 Real LLM functions.** `generate_alert` and `classify_reply` with system prompts in `config/prompts/*.md` (LLM-01–05): at most 25 words, always ends with a question, never asks the driver to look at or touch anything, never encourages driving on when drowsy, never blames. A post-filter checks length, the final question mark and banned phrases, and returns `None` on failure.
- [ ] **VOICE-T10 Templates.** `config/prompts/templates.yaml`: for each alert type and language, at least 3 variants. The lead uses them as the fallback; `prerender.py` turns them into audio.
- [ ] **VOICE-T11 Reply test set.** Collect 30 or more ways of saying yes, no, tired and help in each language, recorded by all four teammates. Text-level tests in `tests/voice/test_intents.py`; an audio-level script for the recordings. *Done when:* 90% or more are classified correctly (VR-03).
- [ ] **VOICE-T12 Latency report.** Log each stage (TTS start, listen, STT, classify) and print medians. Target under 1.5 s median from end of speech to intent (NF-03).
- [ ] **CP2 gate:** F4 acceptance criteria pass; tag `cp2`.

### Before CP3 (Sat 22:00): road rage audio, trivia, check-ins

- [ ] **VOICE-T13 Swear detection.** Continuous transcription of 3 s chunks, only while someone is speaking and outside reply windows; match `config/lexicons/profanity_ro.txt` and `profanity_en.txt` on word boundaries, ignoring diacritics; publish `swear_count_60s`; drop each transcript right after counting (RR-02). *Done when:* 10 phrases with swear words are counted and 10 without are not.
- [ ] **VOICE-T14 Trivia and fun facts.** `generate_engagement`: a question answerable in 1–3 words, or a short fact that ends with a question, tied to the region in `ctx` when given. `grade_answer` accepts close answers ("Paris", "it's Paris", "Parisul"). Offline bank `config/trivia_fallback.json` with 50 items in each language: question, accepted answers, fact (ZO-03, ZO-04, ZO-07).
- [ ] **VOICE-T15 Check-in and calm-down prompts.** Prompt and examples for the check-in question (CI-05) and the calm road-rage line (RR-06): warm, short, no lecturing.
- [ ] **VOICE-T16 Language.** Switch voice, STT language and prompts from `control.trip` `lang`.
- [ ] **CP3 gate:** tag `v0.9-must`.

### Before CP4 (Sun 11:00): extras and robustness

- [ ] **VOICE-T17 Trip summary.** `summarize_trip` prompt (PT-01): time driven, alerts by type, response times, one or two concrete suggestions.
- [ ] **VOICE-T18 Angry-voice model (stretch).** Only if T13 is solid: embeddings from a small pretrained audio model plus logistic regression trained on CREMA-D; publish `anger_voice_p`.
- [ ] **VOICE-T19 Noise test.** Test in the real demo room with a headset or directional mic; add a push-to-talk key as a demo-only fallback.
- [ ] **CP4 gate:** tag `v1.0`.

### Before CP5 (Sun 12:30)

- [ ] **VOICE-T20 Rehearsal support.** Re-render all audio, check offline mode, and keep the mic setup identical for both rehearsals.

## 6. Definition of done for Voice & LLM

Your part is done when every alert is spoken naturally in Romanian or English, every reasonable reply is understood within about 1.5 s, and nothing breaks when the internet does.

- [ ] Voice loop meets VR-01–07; 90% or more of the reply test set classified correctly.
- [ ] All five `backend/llm` functions meet LLM-01–05, return `None` on timeout, and never raise.
- [ ] Templates (3 or more per alert type and language) and pre-rendered audio exist; offline mode tested with Wi-Fi off.
- [ ] Swear and shout detection meet RR-02 and RR-03; no transcripts or audio are stored outside reply logs.
- [ ] Trivia bank of 50 items per language; grading accepts close answers.
- [ ] Latency medians printed for the README; `--fake` mode still works for the others.
