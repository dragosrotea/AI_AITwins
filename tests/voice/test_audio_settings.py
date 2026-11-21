from backend.audio.devices import AudioSettings


def test_legacy_audio_settings_keep_cabin_defaults(tmp_path):
    config = tmp_path / "thresholds.yaml"
    config.write_text(
        "voice:\n  input_device: null\n  output_device: 2\n  ring_buffer_s: 8\n",
        encoding="utf-8",
    )

    settings = AudioSettings.load(config)
    assert settings == AudioSettings(None, 2, 8)
    assert (settings.cabin_baseline_s, settings.cabin_shout_db, settings.cabin_shout_s) == (
        60.0,
        15.0,
        1.0,
    )


def test_audio_settings_read_overrides_from_voice_section(tmp_path):
    config = tmp_path / "thresholds.yaml"
    config.write_text(
        "voice:\n"
        "  input_device: 1\n"
        "  output_device: null\n"
        "  ring_buffer_s: 10\n"
        "  cabin_baseline_s: 45\n"
        "  cabin_shout_db: 17\n"
        "  cabin_shout_s: 0.5\n",
        encoding="utf-8",
    )

    assert AudioSettings.load(config) == AudioSettings(
        1, None, 10, cabin_baseline_s=45, cabin_shout_db=17, cabin_shout_s=0.5
    )
