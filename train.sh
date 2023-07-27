#!/usr/bin/env bash
set -euo pipefail

exec accelerate launch sdxl_train.py --config_file ./wdxl_train.toml
