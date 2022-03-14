#!/bin/bash

check_project_root() {
   if ! [ "$(pwd)" = "${ROOT}" ]; then
      echo "You must be in the root directory ${ROOT}"
      exit 1
   fi
}

# PGD hyperparameters
export EPSILON=350
export STEPS=50
export ALPHA=40

# Docker images
export AAECAPTCHA_IMG="aaecaptcha_ds_041_gpu"
export DS_IMG=${AAECAPTCHA_IMAGE}
export OPENSEQ2SEQ_IMG="nvcr.io/nvidia/tensorflow:19.05-py3"
export KALDI_IMG="kaldiasr/kaldi:gpu-latest"
export LINGVO_IMG="tensorflow:lingvo"