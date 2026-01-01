#!/bin/bash

set -ueo pipefail

# extract mp3 audio from video/audio for transcription

video="${1:-video.mp4}"
out=$(dirname "${video}")

stem="${2:-audio}"
audio="${out}"/"${stem}.mp3"

echo "${video} -> ${audio}"


ffmpeg -i "${video}" -vn -ar 16000 -ac 1 -b:a 48k -c:a libmp3lame "${audio}"

ls -sh "${audio}
