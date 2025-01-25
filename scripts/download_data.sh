#!/usr/bin/env bash

mkdir -p data
cd data
uv run kaggle competitions download -c ai-mathematical-olympiad-progress-prize-2
unzip -o ai-mathematical-olympiad-progress-prize-2.zip
rm -rf ../kaggle_evaluation
mv kaggle_evaluation ..
