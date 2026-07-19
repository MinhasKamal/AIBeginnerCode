# Start Remote Server
start new jupyter server.

# Environment Setup

```
screen -r
conda env list
conda create -p .conda/envs/diffusion -y
conda env list
conda activate diffusion
conda install pip
pip install -U diffusers transformers accelerate sentencepiece protobuf torchvision matplotlib
```

# Hugging Face Token
Go to https://huggingface.co/settings/tokens > Click New read token > copy to clipboard for use in code

Or,
```
pip install -U huggingface_hub
hf auth login
```
and follow the instructions

Then go to https://huggingface.co/black-forest-labs/FLUX.2-dev and agree to the terms for access.

# Run code

```
cd /workspace/minhas/diffusion/
conda activate diffusion
python fluxV2.py
```

