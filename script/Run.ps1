param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$dataRefined = "./assets/data_refined.txt",
    [string]$metadataRefined = "./assets/metadata_refined.txt",
    [string]$refinementInfo = "./assets/refinement_info.txt",
    [string]$logCpuBrf = "./result/cpu-brf.log",
    [string]$logCpuLin = "./result/cpu.log",
    [string]$logGpu = "./result/gpu.log",
    [string]$logGpuExt = "./result/gpu_ext.log",
    [string]$logCpuLinExt = "./result/cpu_ext.log",
    [int]$l = 5, # sequence length
    [int]$n = 20, # number of sequences
    [double]$lf = 0.6 # load factor
)

# generate data
Write-Output "Generating data"
bin/generator.exe $metadata $data $l $n

# eliminate duplicates
Write-Output "`nRemoving duplicates. Saving output to $($refinementInfo), $($metadataRefined), $($dataRefined)"
bin/remove_duplicates.exe $metadata $data $refinementInfo $metadataRefined $dataRefined

# compute results - linear on CPU
Write-Output "`nCPU std::unordered_map. Saving output to $($logCpuLin)"
bin/hammingcpu.exe $metadataRefined $dataRefined $logCpuLin
Write-Output "Process returned $($LASTEXITCODE)"

# compute results - linear on GPU w/ separate chaining hash table
Write-Output "`nSeparate chaining hash table on GPU. Saving output to $($logGpu)"
bin/hamming.exe $metadataRefined $dataRefined $lf $logGpu
Write-Output "Process returned $($LASTEXITCODE)"

# optional part
$validateResults = Read-Host "Validate results? (this will take a while) [y/n]"
if($validateResults -ne "y") {
    Break
}

# extend with duplicates
Write-Output "`nExtending CPU results to include duplicates. Saving output to $($logCpuLinExt)"
bin/extend_to_duplicates.exe $logCpuLin $refinementInfo $logCpuLinExt
Write-Output "`nExtending GPU results to include duplicates. Saving output to $($logGpuExt)"
bin/extend_to_duplicates.exe $logGpu $refinementInfo $logGpuExt

# validate results
Write-Output "`nPerforming brute-force claculations on CPU"
bin/bruteforcecpu.exe $metadata $data $logCpuBrf

Write-Output "`nComparing results from CPU brute-force and CPU std::unordered_map"
bin/compare_results.exe $logCpuBrf $logCpuLinExt

Write-Output "`nComparing results from CPU std::unordered_map and GPU hash table"
bin/compare_results.exe $logCpuLinExt $logGpuExt