#!/usr/bin/env bash

mkdir -p data
cd data
uv run kaggle competitions download -c ai-mathematical-olympiad-progress-prize-2
unzip ai-mathematical-olympiad-progress-prize-2.zip
mv kaggle_evaluation ../aimo2  # move inference server to src
