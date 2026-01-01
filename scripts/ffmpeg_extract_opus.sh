#!/bin/bash

set -ueo pipefail

# extract opus audio from video/audio for transcription

video="${1:-video.mp4}"
out=$(dirname "${video}")

stem="${2:-audio}"
audio="${out}"/"${stem}.opus"

echo "${video} -> ${audio}"

ffmpeg -i "${video}" -vn -ar 16000 -ac 1 -b:a 32k -c:a libopus "${audio}"

ls -sh ${audio}
