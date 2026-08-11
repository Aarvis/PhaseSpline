#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

ENV_NAME="${ENV_NAME:-orpheus}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.18}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
MINICONDA_DIR="${MINICONDA_DIR:-${HOME}/miniconda3}"
MINICONDA_INSTALLER_URL="${MINICONDA_INSTALLER_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${ENV_NAME}"
echo "Python   : ${PYTHON_VERSION}"
echo "PyTorch  : ${PYTORCH_INDEX_URL}"
echo "Miniconda: ${MINICONDA_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Installing Miniconda..."
  if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: neither wget nor curl is available to download Miniconda." >&2
    exit 1
  fi
  INSTALLER_PATH="$(mktemp /tmp/miniconda-installer-XXXXXX.sh)"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${INSTALLER_PATH}" "${MINICONDA_INSTALLER_URL}"
  else
    curl -fsSL "${MINICONDA_INSTALLER_URL}" -o "${INSTALLER_PATH}"
  fi
  bash "${INSTALLER_PATH}" -b -p "${MINICONDA_DIR}"
  rm -f "${INSTALLER_PATH}"
  export PATH="${MINICONDA_DIR}/bin:${PATH}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is still unavailable after Miniconda installation attempt." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
  torch==2.5.1 \
  torchvision==0.20.1

python -m pip install \
  numpy==2.2.6 \
  scipy==1.15.3 \
  pyarrow==20.0.0 \
  pyyaml==6.0.3 \
  tqdm==4.68.3 \
  pillow==12.1.1 \
  opencv-python==5.0.0.93 \
  transformers==4.57.1 \
  safetensors==0.6.2 \
  huggingface-hub==1.27.0 \
  imageio-ffmpeg==0.6.0 \
  matplotlib==3.10.7 \
  wandb==0.25.1

conda install -c conda-forge ffmpeg -y

python -m pip install -e "${REPO_ROOT}/lehome_robot_sim_embedding"
python -m pip install -e "${REPO_ROOT}/lehome_human_spline_generation"

echo
echo "Sanity check"
python - <<'PY'
import torch
import torchvision
import numpy
import scipy
import pyarrow
import yaml
import tqdm
import PIL
import cv2
import transformers
import safetensors
import huggingface_hub
import imageio_ffmpeg

print("python imports: OK")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pyarrow:", pyarrow.__version__)
print("pyyaml:", yaml.__version__)
print("tqdm:", tqdm.__version__)
print("pillow:", PIL.__version__)
print("opencv:", cv2.__version__)
print("transformers:", transformers.__version__)
print("safetensors:", safetensors.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("imageio_ffmpeg:", imageio_ffmpeg.__version__)
PY

echo
echo "Environment setup complete."
echo
echo "Notes:"
echo "1) If you will run DTW-mapping/extract_dinov3_embeddings.py, you must authenticate with Hugging Face:"
echo "   hf auth login"
echo
echo "2) If torch.cuda.is_available() is false, check NVIDIA driver / CUDA compatibility on the remote machine."
echo
echo "3) The following repo areas are covered by this environment:"
echo "   - DTW-mapping"
echo "   - DTW-Transfer"
echo "   - human-spline-localizer"
echo "   - human-to-robot-local-spline-translator"
echo "   - lehome_human_spline_generation"
echo "   - lehome_robot_sim_embedding"
echo "   - lehome_robot_sim_spline_dataset_prep"
