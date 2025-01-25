#!/usr/bin/env bash

uv run ruff check --extend-select I --fix
uv run ruff format
