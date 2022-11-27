param (
    [string]$metadata = "./assets/metadata.txt",
    [string]$data = "./assets/data.txt",
    [string]$dataRefined = "./assets/data_refined.txt",
    [string]$metadataRefined = "./assets/metadata_refined.txt",
    [string]$refinementInfo = "./assets/refinement_info.txt",
    [string]$logCpuBrf = "cpu-brf.log",
    [string]$logCpuLin = "./result/cpu.log",
    [string]$logGpuSC = "./result/gpu_sc.log",
    [string]$logGpuBC = "./result/gpu_bc.log",
    [int]$l = 5, # sequence length
    [int]$n = 20, # number of sequences
    [double]$lf = 0.6 # load factor
)

# generate data
Write-Output "Generate data"
bin/generator.exe $metadata $data $l $n

# eliminate duplicates
Write-Output "`nRemove duplicates"
bin/remove_duplicates.exe $metadata $data $refinementInfo $metadataRefined $dataRefined

# compute results - linear on CPU
Write-Output "`nCPU unordered_map"
bin/hammingcpu.exe $metadataRefined $dataRefined $logCpuLin
Write-Output "Process returned $($LASTEXITCODE)"

# compute results - linear on GPU w/ separate chaining hash table
Write-Output "`nSeparate chaining hash table on GPU"
bin/hamming.exe $metadataRefined $dataRefined $lf $logGpuSC
Write-Output "Process returned $($LASTEXITCODE)"

# compute results - linear on GPU w/ bucketed cuckoo hashing
Write-Output "`nBucketed cuckoo hashing hash table on GPU"
bin/hamming4.exe $metadataRefined $dataRefined $lf $logGpuBC
Write-Output "Process returned $($LASTEXITCODE)"

# validate results
Write-Output "`nCompare CPU multimap vs GPU SC hash table"
bin/compare_results.exe $logCpuLin $logGpuSC

Write-Output "`nCompare CPU multimap vs GPU BC hash table"
bin/compare_results.exe $logCpuLin $logGpuBC