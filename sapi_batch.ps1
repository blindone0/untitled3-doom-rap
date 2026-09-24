# Batch speech synthesis with the Windows built-in voices (System.Speech / SAPI desktop voices).
# One process synthesizes many items, which is what makes per-word synthesis practical.
#   powershell -NoProfile -ExecutionPolicy Bypass -File sapi_batch.ps1 -Spec spec.json [-Voice "Microsoft Zira Desktop"]
# spec.json: [{"text": "static", "rate": 2, "out": "C:\\...\\abc.wav"}, ...]   rate is the SAPI rate, -10..10
param(
    [string]$Spec,
    [string]$Voice = "Microsoft Zira Desktop"
)
Add-Type -AssemblyName System.Speech
# NB: in Windows PowerShell 5.1 ConvertFrom-Json emits a JSON array as ONE object, so @(...) around it would wrap the
# whole array in a single element; assign it and let foreach enumerate instead.
$items = ConvertFrom-Json ([System.IO.File]::ReadAllText($Spec, [System.Text.Encoding]::UTF8))
$n = 0
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($Voice)
# the desktop voices are native 22050 Hz mono
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($it in $items) {
    $synth.Rate = [int]$it.rate
    $synth.SetOutputToWaveFile([string]$it.out, $fmt)
    if ($it.PSObject.Properties.Name -contains 'ssml' -and $it.ssml) {
        $synth.SpeakSsml([string]$it.ssml)   # used to force the weak form of "the" / "a", which are wrong in isolation
    } else {
        $synth.Speak([string]$it.text)
    }
    $synth.SetOutputToNull()
    $n++
}
try { $synth.Dispose() } catch {}
"synthesized $n"
