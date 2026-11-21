import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from backend.llm import client
from backend.voice import evaluate_replies
from backend.voice.stt import Transcription


async def test_evaluation_reports_missing_and_provider_failures(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,lang,speaker_id,expected_intent\n"
        "yes.wav,en,s1,OK\n"
        "help.wav,en,s2,HELP\n"
        "missing.wav,en,s3,NOT_OK\n"
        "bad.wav,en,s4,NOT_OK\n"
        "archived.wav,ro,s1,HELP\n",
        encoding="utf-8",
    )
    for name in ("yes.wav", "help.wav"):
        with wave.open(str(tmp_path / name), "wb") as clip:
            clip.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            clip.writeframes(bytes(4800))
    (tmp_path / "bad.wav").write_bytes(b"invalid")

    transcribe = Mock(
        side_effect=[Transcription("Yes.", 1), Transcription("Please get an ambulance.", 1)]
    )
    model = Mock(return_value=SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(evaluate_replies, "SpeechToText", model)
    provider = AsyncMock(return_value=None)
    monkeypatch.setattr(client, "request_text", provider)

    assert await evaluate_replies.evaluate(evaluate_replies.load_manifest(manifest)) == 1
    output = capsys.readouterr().out
    assert "MISSING en s3:" in output
    assert "AUDIO ERROR en s4 bad.wav:" in output
    assert "PROVIDER FALLBACK FAILED en s2 help.wav:" in output
    assert "expected=HELP actual=UNCLEAR path=llm transcript='Please get an ambulance.'" in output
    assert "s2: listed=1 scored=1 correct=0 accuracy=0.0%" in output
    assert "s1: listed=1 scored=1 correct=1 accuracy=100.0%" in output
    assert "English overall: listed=4 scored=2 correct=1 accuracy=50.0%" in output
    assert "archived.wav" not in output
    assert "T11 gate: PENDING. 4 speakers × 32 clips = 128; 116 correct required." in output
    model.assert_called_once_with(offline=True)
    provider.assert_awaited_once()
    assert [call.args[1] for call in transcribe.call_args_list] == ["en", "en"]
