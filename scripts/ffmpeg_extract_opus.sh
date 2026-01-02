#!/bin/bash

set -ueo pipefail

# extract opus audio from input_path/audio for transcription

input_path="${1:-audio.mp3}"
# use input dir as work dir
work_dir=$(dirname "${input_path}")
input_name=$(basename "${input_path}")
input_stem="${input_name%.*}"

# specify output stem, or fallback to input stem
output_stem="${2:-${input_stem}}"
output_path="${work_dir}"/"${output_stem}.opus"

echo "${input_path} -> ${output_path}"

ffmpeg -i "${input_path}" -vn -ar 16000 -ac 1 -b:a 32k -c:a libopus "${output_path}"

ls -sSh1 ${work_dir}
