# LeHome Spline — ICRA 2027

This repository is organized as a collection of independent components. The current visual episode spline pipeline is entirely contained in:

```text
lehome_spline_generation/
```

That component owns its model code, dataset validation, training, embedding export, spline fitting, configuration, tests, dependency metadata, documentation, and generated `outputs/` directory. See [lehome_spline_generation/README.md](lehome_spline_generation/README.md) for its architecture and commands.

```powershell
cd D:\LeHome-Challenge\Lehome-Spline-ICRA2027\lehome_spline_generation
python -m pip install -e .
python -m pytest
```

Future project components should be added as sibling folders rather than mixed into `lehome_spline_generation/`.


hf upload huggingaccounttest/ICRA2027_ROBOT_Spline_Translator_n10_2_garment E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180 --repo-type=dataset


hf upload huggingaccounttest/ICRA2027_ROBOT_Spline_VLA /mnt/lehome_data/icra2027_spline --repo-type=model

$env:HF_DEBUG="1"
$env:HF_HUB_VERBOSITY="debug"

hf upload huggingaccounttest/ICRA2027_ROBOT_Spline_Translator_n10_2_garment `
"E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180" `
--repo-type=dataset


$env:HF_DEBUG="1"
$env:HF_HUB_VERBOSITY="debug"

hf upload huggingaccounttest/ICRA2027_Human_Spline_Fitted_2_garment "D:/pretrain_lehome_all_garment_data_z180" --repo-type=dataset


hf download huggingaccounttest/ICRA2027_ROBOT_Spline_Translator_n10_2_garment `
  --repo-type dataset `
  --local-dir "E:\Lehome-Dataset\updated_spline_sim_dataset" `
  --force-download


    TrainConfig(
        # Spline-conditioned pi0.5 fine-tune: current joint state + predicted robot spline, no image prefix.
        name="pi05_lehome_robot_spline_joint_delta_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            # Keep the base pi0.5 action width so checkpoint restore stays shape-compatible.
            # LeHome's real 12D actions are handled by the data/output transforms and padded
            # to the model width via PadStatesAndActions.
            action_dim=32,
            discrete_state_input=True,
            robot_spline=pi0_config.RobotSplineConfig(
                enabled=True,
                use_image_prefix=False,
                control_count=13,
                degree=3,
                control_point_dim=2048,
                model_dim=512,
                num_layers=2,
                num_heads=8,
                ffn_dim=2048,
                width_fourier_bands=8,
                width_hidden_dim=512,
                rope_base=10000.0,
            ),
        ),
        num_workers=16,
        run_val=False,
        checkpoint_strategy="manual",
        data=LeRobotLehomeRobotSplineDataConfig(
            repo_id="local/ICRA2027_ROBOT_Spline_Translator_n10_2_garment",
            assets=AssetsConfig(asset_id="pi05_lehome_robot_spline_joint_delta_finetune"),
            base_config=DataConfig(
                prompt_from_task=True,
                robot_spline_sidecar_root=(
                    "/ephemeral/.hf_home/lerobot/local/ICRA2027_ROBOT_Spline_Translator_n10_2_garment/embeddings/robot_sim_multiview_vae_joint_full_visual_epoch/predicted_robot_local_splines_default_run_n010"
                ),
                robot_spline_expand_pairings=True,
                robot_spline_sidecar_required=True,
            ),
            use_delta_joint_actions=True,
            action_dim=12,
            forced_prompt="fold the garment on the table",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-4,
            decay_steps=14000,
            decay_lr=5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|robot_spline_adapter).*",
        ),
        freeze_filter=nnx_utils.PathRegex("PaliGemma/img/.*"),
        non_adapter_lr_multiplier=0.5,
        adapter_param_regex=".*robot_spline_adapter.*",
        num_train_steps=14000,
        batch_size=40,
        log_interval=20,
        save_steps=(3400, 6800,),
        # save_interval=5000,
    ),