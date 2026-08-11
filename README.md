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

$env:HF_DEBUG="1"
$env:HF_HUB_VERBOSITY="debug"

hf upload huggingaccounttest/ICRA2027_ROBOT_Spline_Translator_n10_2_garment `
"E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180" `
--repo-type=dataset


$env:HF_DEBUG="1"
$env:HF_HUB_VERBOSITY="debug"

hf upload huggingaccounttest/ICRA2027_Human_Spline_Fitted_2_garment `
"D:/pretrain_lehome_all_garment_data_z180" `
--repo-type=dataset