from __future__ import annotations

import argparse

from prompt_preprocessor.config import load_config, save_resolved_config
from prompt_preprocessor.pipeline import preprocess_prompt_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw/dataset human prompts into a spline prompt bank.")
    parser.add_argument("--config", required=True, help="Path to the prompt preprocessor YAML/JSON config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a config value with KEY=VALUE.")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    output_root = preprocess_prompt_bank(config)
    save_resolved_config(config, output_root / "resolved_prompt_preprocessor_config.yaml")
    print(f"[prompt-preprocessor] prompt_bank_root={output_root}")


if __name__ == "__main__":
    main()

