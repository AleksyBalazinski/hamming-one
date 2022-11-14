param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$log = "cpu.log"
)

./bin/hammingcpu.exe $metadata $data > $log
Write-Output "Process returned $($LASTEXITCODE)"