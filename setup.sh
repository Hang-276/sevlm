# Read this rather than running it — the conda lines are commented out on
# purpose so you pick the environment name yourself. Whatever you name it,
# set CONDA_ENV in local_scripts/self_evolve/paths.sh to match.

# conda create -n vlm-r1 python=3.11
# conda activate vlm-r1

pip install -e ".[dev]"

pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation   # optional; the code runs on sdpa without it
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]
