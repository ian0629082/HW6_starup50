param(
  [string]$TextPath = "narration.txt",
  [string]$OutDir = "assets\sapi_segments",
  [string]$Output = "assets\narration_fast.wav",
  [int]$Rate = 3
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() |
  Where-Object { $_.VoiceInfo.Culture.Name -like "zh-*" } |
  Select-Object -First 1

if ($voice) {
  $synth.SelectVoice($voice.VoiceInfo.Name)
}

$synth.Rate = $Rate
$synth.Volume = 100

$paragraphs = Get-Content -Path $TextPath -Encoding UTF8 -Raw
$paragraphs = $paragraphs -split "(\r?\n){2,}" | Where-Object { $_.Trim().Length -gt 0 }

$manifest = @()
for ($i = 0; $i -lt $paragraphs.Count; $i++) {
  $text = $paragraphs[$i].Trim()
  $file = Join-Path $OutDir ("segment_{0:D2}.wav" -f ($i + 1))
  $synth.SetOutputToWaveFile((Resolve-Path -LiteralPath (Split-Path -Parent $file)).Path + "\" + (Split-Path -Leaf $file))
  $synth.Speak($text)
  $synth.SetOutputToNull()
  $manifest += [pscustomobject]@{
    index = $i + 1
    file = $file
    text = $text
  }
}

$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $OutDir "manifest.json") -Encoding UTF8

Write-Host ("Generated {0} narration segments with rate {1}." -f $paragraphs.Count, $Rate)
