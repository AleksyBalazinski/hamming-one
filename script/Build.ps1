make generator
if($LASTEXITCODE -ne 0){
    Break
}

make remove_duplicates
if($LASTEXITCODE -ne 0){
    Break
}

make extend_to_duplicates
if($LASTEXITCODE -ne 0){
    Break
}

make bruteforcecpu
if($LASTEXITCODE -ne 0){
    Break
}

make compare_results
if($LASTEXITCODE -ne 0){
    Break
}

make hammingcpu
if($LASTEXITCODE -ne 0){
    Break
}

make hamming
if($LASTEXITCODE -ne 0){
    Break
}