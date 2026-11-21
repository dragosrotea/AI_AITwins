# English T11 recording and evaluation

Have each of the four teammates record the same 32 English phrases in
`confirmation_replies.csv`. Use anonymous IDs `s1` through `s4`, one phrase
per WAV, and a natural voice with a short pause before and after the phrase.
The full set is 128 recordings and requires at least 116 correct to reach 90%.
Keep WAVs outside the repository. Existing English WAVs can be reused. Keep
archived Romanian recordings and the old manifest in place.

The repository phrase catalog now contains only the English set. The evaluator
still skips Romanian rows in older local manifests. The shared broker contract
continues to accept legacy language fields and audio keys.

Run from the repository root in Windows PowerShell:

```powershell
$python = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $python -X utf8 -B -m pytest -q -p no:cacheprovider tests/voice/test_intents.py

$recordingRoot = Join-Path $env:LOCALAPPDATA 'DriveMate\voice-t11'
$phrases = Import-Csv .\tests\voice\confirmation_replies.csv -Encoding UTF8 |
    Where-Object { $_.lang -eq 'en' }
$manifest = foreach ($speaker in 's1', 's2', 's3', 's4') {
    New-Item -ItemType Directory -Force -Path (Join-Path $recordingRoot $speaker) | Out-Null
    foreach ($phrase in $phrases) {
        [pscustomobject]@{
            path            = "$speaker\$($phrase.phrase_id).wav"
            lang            = 'en'
            speaker_id      = $speaker
            expected_intent = $phrase.intent
            text            = $phrase.text
        }
    }
}
$manifestPath = Join-Path $recordingRoot 'manifest-en.csv'
$manifest | Export-Csv $manifestPath -NoTypeInformation -Encoding UTF8
& $python -X utf8 -B -m backend.voice.evaluate_replies $manifestPath
```

The manifest's `text` column is the recording script. Use cached Whisper
`base`. Score with the configured provider online for unmatched transcripts.
The scorer prints English totals per speaker and overall. Missing or unreadable
clips leave the gate pending. Provider failures count as incorrect and leave
the gate pending until a complete provider-enabled run. Keep console output
local if it contains transcripts.
