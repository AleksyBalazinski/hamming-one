#   Copyright 2023 Aleksy Balazinski
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

New-Item -Path . -Name 'assets' -ItemType 'directory' -Force
New-Item -Path . -Name 'bin' -ItemType 'directory' -Force
New-Item -Path . -Name 'result' -ItemType 'directory' -Force

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