git pull
conda remove -n orpheus --all -y
chmod +x setup_remote_linux_env.sh
PYTHON_VERSION=3.10.18 ./setup_remote_linux_env.sh

hf auth login

hf download huggingaccounttest/ICRA2027_ROBOT_Spline_Translator_n10_2_garment \
  --repo-type dataset \
  --local-dir ./ICRA2027_ROBOT_Spline_Translator_n10_2_garment

hf download huggingaccounttest/ICRA2027_Human_Spline_Fitted_2_garment \
  --repo-type dataset \
  --local-dir ./ICRA2027_Human_Spline_Fitted_2_garment