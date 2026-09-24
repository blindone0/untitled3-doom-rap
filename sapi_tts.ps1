# Windows built-in speech synthesis (System.Speech / SAPI desktop voices) to a wav file, with per-word timings.
# Used by static_vocal.py --engine sapi.  Text is read from a UTF-8 file (no quoting trouble).
#   powershell -NoProfile -ExecutionPolicy Bypass -File sapi_tts.ps1 -TextFile in.txt -Out out.wav -Words out.json [-Voice "Microsoft Zira Desktop"] [-Rate 0] [-Pitch "+0%"]
param(
    [string]$TextFile,
    [string]$Out,
    [string]$Words,
    [string]$Voice = "Microsoft Zira Desktop",
    [int]$Rate = 0,
    [string]$Pitch = "+0%"
)
Add-Type -AssemblyName System.Speech
$text = [System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8).Trim()
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($Voice)
$synth.Rate = $Rate
# the desktop voices are native 22050 Hz mono; keep that format so the SpeakProgress AudioPosition is in real seconds
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($Out, $fmt)
$null = Register-ObjectEvent -InputObject $synth -EventName SpeakProgress -SourceIdentifier sapi_words
if ($Pitch -ne "+0%") {
    $esc = [System.Security.SecurityElement]::Escape($text)
    $ssml = "<speak version=`"1.0`" xmlns=`"http://www.w3.org/2001/10/synthesis`" xml:lang=`"en-US`"><prosody pitch=`"$Pitch`">$esc</prosody></speak>"
    $synth.SpeakSsml($ssml)
} else {
    $synth.Speak($text)
}
$synth.SetOutputToNull()
$list = @()
foreach ($e in (Get-Event -SourceIdentifier sapi_words -ErrorAction SilentlyContinue)) {
    $a = $e.SourceEventArgs
    $list += [pscustomobject]@{ t = [double]$a.AudioPosition.TotalSeconds; w = [string]$a.Text; c = [int]$a.CharacterPosition }
}
try { Unregister-Event -SourceIdentifier sapi_words -ErrorAction SilentlyContinue } catch {}
try { $synth.Dispose() } catch {}
if ($list.Count -eq 1) { $json = "[" + ($list | ConvertTo-Json -Compress) + "]" } else { $json = ($list | ConvertTo-Json -Compress) }
if (-not $json) { $json = "[]" }
[System.IO.File]::WriteAllText($Words, $json, (New-Object System.Text.UTF8Encoding($false)))
