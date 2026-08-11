from pathlib import Path

import yaml

from lehome_spline.config import load_config


def test_relative_output_paths_are_scoped_to_component(tmp_path: Path) -> None:
    component = tmp_path / "spline_component"
    config_dir = component / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": {
                    "root": "outputs/model",
                    "embeddings_dir": "outputs/model/embeddings",
                    "splines_dir": "outputs/model/splines",
                    "bspline_dataset_dir": "outputs/model/bspline-dataset",
                    "bspline_external_dir": "outputs/model/bspline-external",
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert Path(config["output"]["root"]) == component / "outputs" / "model"
    assert Path(config["output"]["embeddings_dir"]) == component / "outputs" / "model" / "embeddings"
    assert Path(config["output"]["splines_dir"]) == component / "outputs" / "model" / "splines"
    assert Path(config["output"]["bspline_dataset_dir"]) == component / "outputs" / "model" / "bspline-dataset"
    assert Path(config["output"]["bspline_external_dir"]) == component / "outputs" / "model" / "bspline-external"
    assert Path(config["_component_root"]) == component


def test_absolute_output_override_is_preserved(tmp_path: Path) -> None:
    component = tmp_path / "spline_component"
    config_dir = component / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text("output:\n  root: outputs/model\n", encoding="utf-8")
    absolute_output = (tmp_path / "external-output").resolve()

    config = load_config(config_path, [f"output.root={absolute_output.as_posix()}"])

    assert Path(config["output"]["root"]) == absolute_output
