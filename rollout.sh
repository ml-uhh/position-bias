#!/bin/bash

# Run rollout experiments for various models and configurations

uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/fineweb-edu/bloom-7b1.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/fineweb-edu/bloom-noresidual.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/fineweb-edu/bloom.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/fineweb-edu/falcon-rw-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/fineweb-edu/mpt-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/fineweb-edu/mpt-30b.yaml

uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/dclm/bloom-7b1.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/dclm/bloom.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/dclm/falcon-rw-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/dclm/mpt-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/dclm/mpt-30b.yaml

uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/wikipedia/bloom-7b1.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/wikipedia/bloom.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/wikipedia/falcon-rw-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/wikipedia/mpt-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/all_content/wikipedia/mpt-30b.yaml

uv run --env-file .env -m src.rollout.main config/rollout/alibi/mean_content/bloom-7b1.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/mean_content/bloom-noresidual.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/mean_content/bloom.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/mean_content/falcon-rw-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/mean_content/mpt-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/mean_content/mpt-30b.yaml

uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/bloom-7b1-noresidual.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/bloom-7b1.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/bloom-noresidual.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/bloom.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/falcon-rw-7b-noresidual.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/falcon-rw-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/mpt-7b-noresidual.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/mpt-7b.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/mpt-30b-noresidual.yaml
uv run --env-file .env -m src.rollout.main config/rollout/alibi/no_content/mpt-30b.yaml
