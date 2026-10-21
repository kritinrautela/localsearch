#!/bin/zsh
# Incremental refresh: new/changed/removed files, OCR, embeddings, media. Embedding pauses on battery.
cd ~/localsearch
mkdir -p logs
pgrep -qf "python indexer\.py$" && { echo "a full refresh is already running"; exit 0; }   # quick updates just wait for the lock
{ caffeinate -s /usr/bin/time -l nice -n 10 .venv/bin/python indexer.py; echo "DONE exit=$?"; } > logs/refresh.out 2>&1
.venv/bin/python codesearch.py --index > logs/code.out 2>&1
