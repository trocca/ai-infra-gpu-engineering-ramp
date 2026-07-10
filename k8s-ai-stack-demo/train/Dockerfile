# Layer 10 — the container image. The second of the three things the developer writes.
#
# The NGC PyTorch image is a monthly, ABI-matched, pre-validated stack:
# NVIDIA-built PyTorch + CUDA + cuDNN + NCCL + TransformerEngine + Apex + DALI.
# Tag YY.MM-pyN is a CALENDAR version (26.04 = April 2026 build), not the
# PyTorch version. Re-pin to the current month before a live demo.
FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /workspace
COPY train.py /workspace/train.py

# No CMD/ENTRYPOINT on purpose: the TrainJob CRD supplies the command
# (torchrun ... train.py), keeping the image generic and the intent in YAML.
