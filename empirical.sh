#!/bin/bash

# Run empirical experiments for various models and configurations

# 7B-9B Models
# ALiBi
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/falcon-rw-7b-64.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/falcon-rw-7b-256.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/falcon-rw-7b-512.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/falcon-rw-7b-1024.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/falcon-rw-7b-2048.yaml

# ALiBi
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-7b1-64.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-7b1-256.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-7b1-512.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-7b1-1024.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-7b1-2048.yaml

# ALiBi
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-7b-64.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-7b-256.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-7b-512.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-7b-1024.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-7b-2048.yaml

# 70B+ Models

# ALiBi
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-30B-64.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-30B-256.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-30B-512.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-30B-1024.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/mpt-30B-2048.yaml

# ALiBi
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-64.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-256.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-512.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-1024.yaml
yes | uv run --env-file .env -m src.main ./config/empirical/alibi/bloom-2048.yaml
