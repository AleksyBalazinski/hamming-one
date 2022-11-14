param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$log = "gpu.log"
)

./bin/hamming.exe $metadata $data > $log
Write-Output "Process returned $($LASTEXITCODE)"