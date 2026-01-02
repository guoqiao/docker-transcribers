#!/bin/bash

set -ueo pipefail

file=${1:-data/audio.mp3}
format=${2:-text}
language=${3:-zh}

curl -sS -X POST "http://localhost:8000/v1/audio/transcriptions" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@${file}" \
    -F "response_format=${format}" \
    -F "language=${language}" \
    | jq -r
