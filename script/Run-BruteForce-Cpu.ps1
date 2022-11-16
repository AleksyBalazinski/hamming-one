param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$log = "cpu-brf.log"
)

./bin/bruteforcecpu.exe $metadata $data | Out-File -encoding ascii $log
Write-Output "Process returned $($LASTEXITCODE)"