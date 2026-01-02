#!/bin/bash

set -ueo pipefail

file=${1:-data/bcn_weather.mp3}
format=${2:-}
language=${3:-}
model=${4:-}

curl -sS https://api.openai.com/v1/audio/transcriptions \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    -H "Content-Type: multipart/form-data" \
    -F model="whisper-1" \
    -F language="${language}" \
    -F response_format="${format}" \
    -F file="@${file}"
