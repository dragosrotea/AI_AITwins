"""Alert Manager v1 (LEAD-T10): every path, driven by a fake clock."""

from __future__ import annotations

from typing import Any

import pytest

from backend.config import load_config
from backend.core.alerts import ESCALATE_TEXT, REPEAT_TEXT, Alert, AlertManager
from backend.schemas import validate

RATE_CFG = load_config("default")["core"]["alerts"]  # real AL-02 limits (LEAD-T14 tests)
# Queue and reply tests submit alerts seconds apart: run them without the rate limits.
CFG = {**RATE_CFG, "t1_min_interval_s": 0, "noncritical_min_interval_s": 0, "cooldown_s": {}}
GAP = CFG["voice_gap_s"]
LISTEN = CFG["listen_s"] + CFG["reply_grace_s"]
FIRST_DEADLINE = CFG["speak_max_s"] + LISTEN


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Rig:
    """An AlertManager wired to a fake clock, a message recorder and recording hooks."""

    def __init__(self, cfg: dict[str, Any] = CFG) -> None:
        self.clock = FakeClock()
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.hooks: list[tuple[str, str, str | None]] = []
        self.suppressed: list[tuple[str, str, str]] = []
        self.am = AlertManager(
            cfg,
            self.publish,
            self.clock,
            on_follow_up=self.hook("follow_up"),
            on_emergency=self.hook("emergency"),
            on_alarm=self.hook("alarm"),
            on_suppress=lambda kind, tier, why: self.suppressed.append((kind, tier, why)),
        )

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)  # everything the manager sends must match the contract
        self.sent.append((topic, data))

    def hook(self, name: str) -> Any:
        def record(alert: Alert, reply: dict[str, Any] | None) -> None:
            self.hooks.append((name, alert.type, reply["intent"] if reply else None))

        return record

    def advance(self, seconds: float, step: float = 0.1) -> None:
        end = self.clock.now + seconds
        while self.clock.now < end:
            self.clock.now = min(end, self.clock.now + step)
            if end - self.clock.now < 1e-9:
                self.clock.now = end  # land exactly on the target, despite float steps
            self.am.tick()

    def says(self) -> list[dict[str, Any]]:
        return [d for t, d in self.sent if t == "voice.say"]

    def statuses(self, alert: Alert | None = None) -> list[str]:
        return [
            d["status"]
            for t, d in self.sent
            if t == "alert.state" and (alert is None or d["alert_id"] == alert.alert_id)
        ]

    def music(self) -> list[tuple[bool, float]]:
        return [(d["duck"], d["level"]) for t, d in self.sent if t == "control.music"]

    def listening(self, alert: Alert) -> None:
        self.am.on_voice_status({"alert_id": alert.alert_id, "state": "listening"})

    def reply(self, alert: Alert, intent: str, transcript: str = "yes") -> None:
        self.am.on_voice_reply(
            {
                "alert_id": alert.alert_id,
                "transcript": transcript,
                "intent": intent,
                "latency_ms": 420,
                "path": "keyword",
            }
        )


@pytest.fixture
def rig() -> Rig:
    return Rig()


def submit(rig: Rig, type: str = "distraction", tier: str = "T2", **kw: Any) -> Alert:
    alert = rig.am.submit(type, tier, f"{type} text?", short_text=f"{type} short?", **kw)
    assert alert is not None
    return alert


# Happy paths


def test_ok_reply_closes_and_restores_music(rig: Rig) -> None:
    alert = submit(rig)
    assert rig.statuses() == ["queued", "speaking"]
    say = rig.says()[0]
    assert say["alert_id"] == alert.alert_id and say["tier"] == "T2"
    assert say["text"] == "distraction text?" and say["listen_s"] == CFG["listen_s"]
    assert rig.music() == [(True, CFG["music_duck_volume"] / 100)]
    rig.listening(alert)
    assert rig.statuses()[-1] == "listening"
    rig.reply(alert, "OK")
    closed = [d for t, d in rig.sent if t == "alert.state"][-1]
    assert closed["status"] == "closed" and closed["intent"] == "OK"
    assert closed["reply"] == "yes" and closed["latency_ms"] == 420
    assert rig.am.active is None
    rig.advance(GAP)
    assert rig.music()[-1] == (False, 1.0)
    assert rig.hooks == []


def test_reply_without_voice_status_still_closes(rig: Rig) -> None:
    """Fake voice sends no voice.status, only a reply."""
    alert = submit(rig)
    rig.reply(alert, "OK")
    assert rig.statuses() == ["queued", "speaking", "closed"]


def test_answer_closes(rig: Rig) -> None:
    alert = submit(rig, "zoning_out", "T1")
    rig.reply(alert, "ANSWER", "Lisbon")
    assert rig.statuses()[-1] == "closed" and rig.hooks == []


def test_not_ok_calls_follow_up(rig: Rig) -> None:
    alert = submit(rig, "drowsiness")
    rig.reply(alert, "NOT_OK", "I'm tired")
    assert rig.statuses()[-1] == "closed"
    assert rig.hooks == [("follow_up", "drowsiness", "NOT_OK")]


def test_help_calls_emergency(rig: Rig) -> None:
    alert = submit(rig, "checkin_due", "T1")
    rig.reply(alert, "HELP", "help")
    assert rig.statuses()[-1] == "closed"
    assert rig.hooks == [("emergency", "checkin_due", "HELP")]


# Misses: repeat once, then escalate (VR-05)


def test_unclear_repeats_once_with_short_text_after_gap(rig: Rig) -> None:
    alert = submit(rig)
    rig.reply(alert, "UNCLEAR", "mmh")
    assert len(rig.says()) == 1  # voice is still busy: wait for the gap
    rig.advance(GAP)
    repeat = rig.says()[1]
    assert repeat["alert_id"] == alert.alert_id and repeat["text"] == "distraction short?"
    assert repeat["tier"] == "T2"
    rig.reply(alert, "OK")
    assert rig.statuses(alert)[-1] == "closed"


def test_second_miss_escalates_one_tier(rig: Rig) -> None:
    alert = submit(rig)  # T2
    rig.reply(alert, "UNCLEAR")
    rig.advance(GAP)
    rig.reply(alert, "NONE", "")
    assert rig.statuses(alert)[-1] == "escalated"
    rig.advance(GAP)
    say = rig.says()[-1]
    assert say["tier"] == "T3" and say["text"] == ESCALATE_TEXT["en"]
    assert rig.music()[-1] == (True, CFG["t3_music_level"])  # T3: music muted
    assert alert.tier == "T3"


def test_t1_escalates_to_t2(rig: Rig) -> None:
    alert = submit(rig, "checkin_due", "T1")
    for _ in range(2):
        rig.reply(alert, "NONE", "")
        rig.advance(GAP)
    assert rig.says()[-1]["tier"] == "T2"


def test_silence_through_t3_sounds_alarm(rig: Rig) -> None:
    """VR-05 acceptance: staying silent through the prompts produces the alarm."""
    alert = submit(rig)  # T2: say, repeat, escalate to T3: say, repeat, then the alarm
    for _ in range(4):
        rig.reply(alert, "NONE", "")
        rig.advance(GAP)
    assert [s["tier"] for s in rig.says()] == ["T2", "T2", "T3", "T3", "T3"]
    assert rig.says()[-1]["audio_key"] == CFG["alarm_audio_key"]  # the alarm clip plays
    assert rig.hooks == [("alarm", "distraction", None)]
    assert rig.am.active is alert  # still waiting for an answer
    assert rig.statuses(alert).count("escalated") == 2


def test_alarm_is_continuous_until_answered(rig: Rig) -> None:
    alert = submit(rig, "microsleep", "T3")
    for _ in range(2):  # say, repeat
        rig.reply(alert, "NONE", "")
        rig.advance(GAP)
    for _ in range(3):  # three alarm rounds, no answer
        assert rig.says()[-1]["audio_key"] == CFG["alarm_audio_key"]
        rig.reply(alert, "NONE", "")
        rig.advance(GAP)
    rig.reply(alert, "OK", "I'm awake")
    assert rig.statuses(alert)[-1] == "closed" and rig.am.active is None
    assert rig.hooks == [("alarm", "microsleep", None)]  # the hook fires once, at the start


def test_alarm_stops_after_its_rounds(rig: Rig) -> None:
    alert = submit(rig, "microsleep", "T3")
    for _ in range(2 + CFG["alarm_rounds"]):
        rig.reply(alert, "NONE", "")
        rig.advance(GAP)
    alarms = [s for s in rig.says() if s.get("audio_key") == CFG["alarm_audio_key"]]
    assert len(alarms) == CFG["alarm_rounds"]
    assert rig.am.active is None  # gives up, so the queue can move on


def test_alarm_rounds_also_time_out(rig: Rig) -> None:
    """Without any voice reply at all, the deadlines drive the alarm too."""
    alert = submit(rig, "microsleep", "T3")
    rig.advance(3 * (FIRST_DEADLINE + GAP + 0.2))
    assert alert.alarm_rounds >= 1
    assert rig.says()[-1]["audio_key"] == CFG["alarm_audio_key"]


def test_silence_after_crash_also_starts_emergency(rig: Rig) -> None:
    alert = submit(rig, "crash_candidate", "T3")
    for _ in range(2):
        rig.reply(alert, "NONE", "")
        rig.advance(GAP)
    assert rig.hooks == [("alarm", "crash_candidate", None), ("emergency", "crash_candidate", None)]


def test_romanian_repeat_without_short_text(rig: Rig) -> None:
    alert = rig.am.submit("phone", "T2", "Lasă telefonul. Ești atent?", lang="ro")
    assert alert is not None
    rig.reply(alert, "UNCLEAR")
    rig.advance(GAP)
    assert rig.says()[-1]["text"] == REPEAT_TEXT["ro"] and rig.says()[-1]["lang"] == "ro"


# Timeouts


def test_no_status_and_no_reply_times_out_from_voice_say(rig: Rig) -> None:
    alert = submit(rig)
    rig.advance(FIRST_DEADLINE - 0.2)
    assert len(rig.says()) == 1
    rig.advance(0.3)
    assert alert.misses == 1 and alert.intent == "NONE"
    rig.advance(GAP)
    assert len(rig.says()) == 2


def test_listening_restarts_the_window(rig: Rig) -> None:
    alert = submit(rig)
    rig.advance(5.0)  # speaking took 5 s
    rig.listening(alert)
    rig.advance(LISTEN - 0.2)
    assert alert.misses == 0
    rig.advance(0.3)
    assert alert.misses == 1


def test_no_timeout_while_waiting_in_the_gap(rig: Rig) -> None:
    alert = submit(rig)
    rig.reply(alert, "UNCLEAR")
    rig.clock.now += 10 * FIRST_DEADLINE  # a long stall must not count as another miss
    rig.am.tick()
    assert alert.misses == 1 and len(rig.says()) == 2


# Queue and priority (AL-01)


def test_one_at_a_time_then_by_tier_then_fifo(rig: Rig) -> None:
    first = submit(rig, "checkin_due", "T1")
    t1 = submit(rig, "zoning_out", "T1")
    t2 = submit(rig, "drowsiness", "T2")
    assert len(rig.says()) == 1 and rig.am.active is first
    assert [a.type for a in rig.am.queue] == ["drowsiness", "zoning_out"]
    rig.reply(first, "OK")
    rig.advance(GAP)
    assert rig.am.active is t2
    rig.reply(t2, "OK")
    rig.advance(GAP)
    assert rig.am.active is t1
    assert (False, 1.0) not in rig.music()  # music stays ducked between back-to-back alerts


def test_t3_preempts_and_the_interrupted_alert_restarts(rig: Rig) -> None:
    t2 = submit(rig, "distraction", "T2")
    rig.listening(t2)
    t3 = submit(rig, "microsleep", "T3")
    assert rig.am.active is t3
    assert rig.says()[-1]["alert_id"] == t3.alert_id  # no gap for T3
    assert rig.statuses(t2)[-1] == "queued"
    rig.reply(t2, "OK")  # a late reply for the interrupted alert is ignored
    assert rig.am.active is t3
    rig.reply(t3, "OK")
    rig.advance(GAP)
    assert rig.am.active is t2 and rig.says()[-1]["text"] == "distraction text?"


def test_t3_does_not_preempt_t3(rig: Rig) -> None:
    first = submit(rig, "microsleep", "T3")
    submit(rig, "crash_candidate", "T3")
    assert rig.am.active is first and len(rig.says()) == 1


def test_t3_preempts_an_alert_waiting_to_repeat(rig: Rig) -> None:
    t2 = submit(rig)
    rig.reply(t2, "UNCLEAR")  # repeat pending in the gap
    t3 = submit(rig, "microsleep", "T3")
    assert rig.am.active is t3
    rig.advance(GAP)
    assert rig.says()[-1]["alert_id"] == t3.alert_id  # the old repeat was dropped


def test_t3_skips_the_gap_after_a_close(rig: Rig) -> None:
    t2 = submit(rig)
    rig.reply(t2, "OK")
    t3 = submit(rig, "microsleep", "T3")
    assert rig.says()[-1]["alert_id"] == t3.alert_id


def test_new_alert_waits_for_the_gap_after_a_close(rig: Rig) -> None:
    first = submit(rig)
    rig.reply(first, "OK")
    second = submit(rig, "phone", "T2")
    assert rig.am.active is None and len(rig.says()) == 1
    rig.advance(GAP)
    assert rig.am.active is second and len(rig.says()) == 2


def test_duplicate_type_is_dropped(rig: Rig) -> None:
    submit(rig, "distraction")
    assert rig.am.submit("distraction", "T2", "again?") is None
    submit(rig, "phone")
    assert rig.am.submit("phone", "T2", "again?") is None


# Ignored input


def test_stale_and_unknown_messages_are_ignored(rig: Rig) -> None:
    alert = submit(rig)
    rig.am.on_voice_reply({"alert_id": "a-other", "intent": "OK"})
    rig.am.on_voice_status({"alert_id": "a-other", "state": "listening"})
    assert rig.am.active is alert and alert.status == "speaking"
    rig.reply(alert, "UNCLEAR")
    rig.reply(alert, "UNCLEAR")  # duplicate during the gap: not a second miss
    assert alert.misses == 1
    rig.am.on_voice_status({"alert_id": alert.alert_id, "state": "idle"})
    assert alert.status == "speaking"


def test_idle_manager_ignores_voice_messages(rig: Rig) -> None:
    rig.am.on_voice_reply({"alert_id": "a-x", "intent": "OK"})
    rig.am.tick()
    assert rig.sent == []


def test_audio_key_only_with_the_original_text(rig: Rig) -> None:
    alert = rig.am.submit("drowsiness", "T2", "Break?", audio_key="drowsiness_en_1")
    assert alert is not None
    assert rig.says()[0]["audio_key"] == "drowsiness_en_1"
    rig.reply(alert, "UNCLEAR")
    rig.advance(GAP)
    assert "audio_key" not in rig.says()[1]  # the repeat uses other words


# Genuine T3 vs escalated T3 (17:12 demo: a microsleep waited 22 s behind an escalation)


def test_genuine_t3_preempts_an_escalated_alert(rig: Rig) -> None:
    distraction = submit(rig)  # T2
    rig.reply(distraction, "NONE", "")
    rig.advance(GAP)
    rig.reply(distraction, "NONE", "")
    rig.advance(GAP)
    assert distraction.tier == "T3" and rig.am.active is distraction  # escalated
    micro = submit(rig, "microsleep", "T3")
    assert rig.am.active is micro and rig.says()[-1]["alert_id"] == micro.alert_id
    assert distraction in rig.am.queue  # resumes after the microsleep
    rig.reply(micro, "OK")
    rig.advance(GAP)
    assert rig.am.active is distraction


# LEAD-T14: rate limits and cooldowns (AL-02), with the real config values


@pytest.fixture
def limited() -> Rig:
    return Rig(RATE_CFG)


def finish(rig: Rig, alert: Alert) -> None:
    rig.reply(alert, "OK")
    rig.advance(GAP)


def test_one_non_critical_prompt_per_interval(limited: Rig) -> None:
    first = submit(limited, "distraction", "T2")
    finish(limited, first)
    assert limited.am.submit("phone", "T2", "Phone?") is None
    assert limited.suppressed == [
        ("phone", "T2", f"one non-critical prompt per {RATE_CFG['noncritical_min_interval_s']} s")
    ]
    limited.advance(RATE_CFG["noncritical_min_interval_s"])
    assert limited.am.submit("phone", "T2", "Phone?") is not None


def test_one_t1_per_five_minutes(limited: Rig) -> None:
    finish(limited, submit(limited, "checkin_due", "T1"))
    limited.advance(RATE_CFG["noncritical_min_interval_s"])  # the 2 min limit has passed
    assert limited.am.submit("zoning_out", "T1", "Trivia?") is None
    assert "one T1 per" in limited.suppressed[-1][2]
    t2 = limited.am.submit("distraction", "T2", "Focus?")
    assert t2 is not None  # a T2 is not held by the T1 limit
    finish(limited, t2)
    limited.advance(RATE_CFG["t1_min_interval_s"])
    assert limited.am.submit("zoning_out", "T1", "Trivia?") is not None


def test_per_type_cooldown(limited: Rig) -> None:
    finish(limited, submit(limited, "drowsiness", "T2"))
    limited.advance(RATE_CFG["noncritical_min_interval_s"])  # global limit passed
    assert limited.am.submit("drowsiness", "T2", "Break?") is None  # 5 min cooldown
    cooldown = RATE_CFG["cooldown_s"]["drowsiness"]
    assert limited.suppressed[-1][2] == f"drowsiness cooldown {cooldown} s"
    limited.advance(RATE_CFG["cooldown_s"]["drowsiness"])
    assert limited.am.submit("drowsiness", "T2", "Break?") is not None


def test_t3_is_never_limited(limited: Rig) -> None:
    first = submit(limited, "microsleep", "T3")
    finish(limited, first)
    assert limited.am.submit("microsleep", "T3", "Awake?") is not None
    finish(limited, limited.am.active)
    finish(limited, submit(limited, "distraction", "T2"))
    assert limited.am.submit("crash_candidate", "T3", "Okay?") is not None  # right after a T2
    assert limited.suppressed == []


def test_repeats_and_escalations_are_not_limited(limited: Rig) -> None:
    alert = submit(limited, "distraction", "T2")
    for _ in range(3):
        limited.reply(alert, "NONE", "")
        limited.advance(GAP)
    assert [s["tier"] for s in limited.says()] == ["T2", "T2", "T3", "T3"]
    assert limited.suppressed == []


def test_suppressed_alerts_are_never_spoken(limited: Rig) -> None:
    finish(limited, submit(limited, "distraction", "T2"))
    before = len(limited.says())
    limited.am.submit("no_face", "T2", "Okay?")
    limited.advance(GAP)
    assert len(limited.says()) == before
    assert all(d["type"] != "no_face" for t, d in limited.sent if t == "alert.state")


def test_demo_profile_shortens_the_limits() -> None:
    demo = load_config("demo")["core"]["alerts"]
    assert demo["noncritical_min_interval_s"] < RATE_CFG["noncritical_min_interval_s"]
    assert demo["t1_min_interval_s"] < RATE_CFG["t1_min_interval_s"]
    assert all(demo["cooldown_s"][k] <= v for k, v in RATE_CFG["cooldown_s"].items())


# LEAD-T23: hold T1 prompts while the road load is HIGH (AL-03, MAP-03)


def road(rig: Rig, load: str) -> None:
    rig.am.on_road({"load": load})


def test_queued_trivia_plays_only_after_the_busy_stretch(rig: Rig) -> None:
    """MAP-03 acceptance: a queued trivia question plays only after the roundabout."""
    first = submit(rig, "distraction", "T2")
    trivia = submit(rig, "zoning_out", "T1")  # waits behind the T2
    road(rig, "HIGH")
    assert rig.statuses(trivia)[-1] == "held"  # visible as held on the dashboard (ZO-06)
    rig.reply(first, "OK")
    rig.advance(GAP)
    assert rig.am.active is None  # nothing else may play: the trivia is held
    road(rig, "LOW")
    assert rig.am.active is trivia and rig.statuses(trivia)[-2:] == ["queued", "speaking"]


def test_new_t1_arrives_held_and_t2_t3_are_never_held(rig: Rig) -> None:
    road(rig, "HIGH")
    checkin = submit(rig, "checkin_due", "T1")
    assert rig.statuses(checkin) == ["held"] and rig.am.active is None
    warning = submit(rig, "phone", "T2")
    assert rig.am.active is warning  # T2 is spoken at once
    rig.reply(warning, "OK")
    rig.advance(GAP)
    micro = submit(rig, "microsleep", "T3")
    assert rig.am.active is micro
    assert rig.statuses(checkin) == ["held"]  # still waiting


def test_a_playing_t1_finishes_when_the_load_rises(rig: Rig) -> None:
    checkin = submit(rig, "checkin_due", "T1")
    road(rig, "HIGH")
    assert rig.am.active is checkin
    rig.reply(checkin, "OK")
    assert rig.statuses(checkin)[-1] == "closed"


def test_medium_load_does_not_hold(rig: Rig) -> None:
    road(rig, "MEDIUM")
    checkin = submit(rig, "checkin_due", "T1")
    assert rig.am.active is checkin


def test_repeated_road_messages_change_nothing(rig: Rig) -> None:
    road(rig, "HIGH")
    trivia = submit(rig, "zoning_out", "T1")
    for _ in range(5):
        road(rig, "HIGH")  # context.road comes at 1 Hz
    assert rig.statuses(trivia) == ["held"]
