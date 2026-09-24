# Batch speech synthesis with the Windows built-in voices (System.Speech / SAPI desktop voices).
# One process synthesizes many items, which is what makes per-line and per-word synthesis practical.
#   powershell -NoProfile -ExecutionPolicy Bypass -File sapi_batch.ps1 -Spec spec.json [-Voice "Microsoft Zira Desktop"]
# spec.json: [{"text": "static", "rate": 2, "out": "C:\\...\\abc.wav", "words": "C:\\...\\abc.words.json"}, ...]
#   rate is the SAPI rate, -10..10; "words" is optional and receives the per-word timings of that item.
# NB: the desktop voices are native 22050 Hz mono, and SpeakProgress reports AudioPosition in NOMINAL-rate time,
# so the caller has to scale the marks by the voice's measured rate factor.
param(
    [string]$Spec,
    [string]$Voice = "Microsoft Zira Desktop"
)
Add-Type -AssemblyName System.Speech
# In Windows PowerShell 5.1 ConvertFrom-Json emits a JSON array as ONE object, so @(...) around it would wrap the
# whole array in a single element; assign it and let foreach enumerate instead.
$items = ConvertFrom-Json ([System.IO.File]::ReadAllText($Spec, [System.Text.Encoding]::UTF8))
$n = 0
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($Voice)
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$null = Register-ObjectEvent -InputObject $synth -EventName SpeakProgress -SourceIdentifier sapi_words
foreach ($it in $items) {
    Get-Event -SourceIdentifier sapi_words -ErrorAction SilentlyContinue | Remove-Event -ErrorAction SilentlyContinue
    $synth.Rate = [int]$it.rate
    $synth.SetOutputToWaveFile([string]$it.out, $fmt)
    if ($it.PSObject.Properties.Name -contains 'ssml' -and $it.ssml) {
        $synth.SpeakSsml([string]$it.ssml)
    } else {
        $synth.Speak([string]$it.text)
    }
    $synth.SetOutputToNull()
    if ($it.PSObject.Properties.Name -contains 'words' -and $it.words) {
        $list = @()
        foreach ($e in (Get-Event -SourceIdentifier sapi_words -ErrorAction SilentlyContinue)) {
            $a = $e.SourceEventArgs
            $list += [pscustomobject]@{ t = [double]$a.AudioPosition.TotalSeconds; w = [string]$a.Text; c = [int]$a.CharacterPosition }
        }
        if ($list.Count -eq 1) { $json = "[" + ($list | ConvertTo-Json -Compress) + "]" }
        elseif ($list.Count -eq 0) { $json = "[]" }
        else { $json = ($list | ConvertTo-Json -Compress) }
        [System.IO.File]::WriteAllText([string]$it.words, $json, (New-Object System.Text.UTF8Encoding($false)))
    }
    $n++
}
try { Unregister-Event -SourceIdentifier sapi_words -ErrorAction SilentlyContinue } catch {}
try { $synth.Dispose() } catch {}
"synthesized $n"
