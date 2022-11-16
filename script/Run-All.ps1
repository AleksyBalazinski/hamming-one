param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$logCpuBrf = "cpu-brf.log",
    [string]$logCpuLin = "cpu.log",
    [string]$logGpuLin = "gpu.log",
    [int]$seqLength = 5,
    [int]$numOfSeqs = 20
)

# generate data
./bin/generator.exe $metadata $data $seqLength $numOfSeqs

# compute results - brute force on CPU
$timeBrf = Measure-Command {
    ./bin/bruteforcecpu.exe $metadata $data | Out-File -encoding ascii $logCpuBrf
    Write-Output "Process returned $($LASTEXITCODE)"
}
Write-Output "Brute force on CPU took $($timeBrf.Milliseconds)"

# compute results - linear on CPU
$timeLinCpu = Measure-Command {
    ./bin/hammingcpu.exe $metadata $data | Out-File -encoding ascii $logCpuLin
    Write-Output "Process returned $($LASTEXITCODE)"
}
Write-Output "Linear on CPU took $($timeLinCpu.Milliseconds)"

# validate results
./bin/CompareResults $logCpuBrf $logCpuLin

# compute results - linear on GPU
$timeLinGpu = Measure-Command {
    ./bin/hamming.exe $metadata $data | Out-File -encoding ascii $logGpuLin
    Write-Output "Process returned $($LASTEXITCODE)"
}
Write-Output "Linear on GPU took $($timeLinGpu.Milliseconds)"

# validate results
./bin/CompareResults $logCpuBrf $logGpuLin