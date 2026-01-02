#!/bin/bash

set -ueo pipefail

file=${1:-data/bcn_weather.mp3}
format=${2:-}
language=${3:-}

curl -sS -X POST "http://localhost:8000/v1/audio/transcriptions" \
    -H "Content-Type: multipart/form-data" \
    -F "language=${language}" \
    -F "response_format=${format}" \
    -F "file=@${file}"
