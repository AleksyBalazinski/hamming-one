param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$log = "cpu.log"
)

./bin/hammingcpu.exe $metadata $data | Out-File -encoding ascii $log
Write-Output "Process returned $($LASTEXITCODE)"