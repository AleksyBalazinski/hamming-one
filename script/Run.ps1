param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$dataRefined = "./assets/data_refined.txt",
    [string]$metadataRefined = "./assets/metadata_refined.txt",
    [string]$refinementInfo = "./assets/refinement_info.txt",
    [string]$logCpuBrf = "./result/cpu-brf.txt",
    [string]$logCpuLin = "./result/cpu.txt",
    [string]$logGpu = "./result/gpu.txt",
    [string]$logGpuExt = "./result/gpu_ext.txt",
    [string]$logCpuLinExt = "./result/cpu_ext.txt",
    [int]$l = 5, # sequence length
    [int]$n = 20, # number of sequences
    [double]$lf = 0.6 # load factor
)

# generate data
Write-Host "Generating $($n) sequences of length $($l). Saving output to $($metadata), $($data)"
bin/generator.exe $metadata $data $l $n

# eliminate duplicates
Write-Host "`nRemoving duplicates. Saving output to $($refinementInfo), $($metadataRefined), $($dataRefined)"
bin/remove_duplicates.exe $metadata $data $refinementInfo $metadataRefined $dataRefined

# compute results - linear on CPU
Write-Host "`nCPU std::unordered_map. Saving output to $($logCpuLin)"
bin/hammingcpu.exe $metadataRefined $dataRefined $logCpuLin
Write-Host "Process returned $($LASTEXITCODE)"

# compute results - linear on GPU w/ separate chaining hash table
Write-Host "`nSeparate chaining hash table on GPU. Saving output to $($logGpu)"
bin/hamming.exe $metadataRefined $dataRefined $lf $logGpu
Write-Host "Process returned $($LASTEXITCODE)"

# optional part
$validateResults = Read-Host "`nValidate results? (this will take a while) [y/n]"
if($validateResults -ne "y") {
    Break
}

# extend with duplicates
Write-Host "`nExtending CPU results to include duplicates. Saving output to $($logCpuLinExt)"
bin/extend_to_duplicates.exe $logCpuLin $refinementInfo $logCpuLinExt
Write-Host "`nExtending GPU results to include duplicates. Saving output to $($logGpuExt)"
bin/extend_to_duplicates.exe $logGpu $refinementInfo $logGpuExt

# validate results
Write-Host "`nPerforming brute-force calculations on CPU"
bin/bruteforcecpu.exe $metadata $data $logCpuBrf

Write-Host "`nComparing results from CPU brute-force with CPU std::unordered_map"
bin/compare_results.exe $logCpuBrf $logCpuLinExt

Write-Host "`nComparing results from CPU std::unordered_map with GPU hash table"
bin/compare_results.exe $logCpuLinExt $logGpuExt