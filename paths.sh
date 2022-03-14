#!/bin/bash
export ROOT=$(pwd)
export DATA_ROOT=${ROOT}/data
export LOG=${ROOT}/logs
export ATTACK_ASR_ROOT=${ROOT}/attack_ASR_models
export LINGVO_ROOT=${ATTACK_ASR_ROOT}/lingvo_asr
export KALDI_ROOT=${ATTACK_ASR_ROOT}/kaldi_asr
export DS2_JASPER_W2LPLUS_ROOT=${ATTACK_ASR_ROOT}/ds2_jasper_w2lplus/OpenSeq2Seq
export DEVICE="gpu"