#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

DS_CHECKPOINT_URL="https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz"
DS_CHECKPOINT_TAR="deepspeech-0.4.1-checkpoint.tar.gz"

cd "${ROOT}"
check_project_root

git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech || exit 1
git checkout v0.4.1
cd ..

if [ ! -d deepspeech-0.4.1-checkpoint ]; then
    [[ ! -f "${DS_CHECKPOINT_TAR}" ]] && wget "${DS_CHECKPOINT_URL}"
    tar -xvf "${DS_CHECKPOINT_TAR}" || exit 1
    rm "${DS_CHECKPOINT_TAR}"
fi

# Download LM for DeepSpeech
wget -O DeepSpeech/data/lm/lm.binary  "https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/lm.binary"
wget -O DeepSpeech/data/lm/trie "https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/trie"


for d in attack_ASR_models logs data/librispeech data/audio_captchas; do
  [ ! -d "${d}" ] &&  mkdir "${d}"
done

docker build --no-cache -t "${AAECAPTCHA_IMG}" - <docker/aaeCaptcha_ds_v041_gpu.dockerfile || exit 1

docker run --gpus all --rm -it \
   -v "${ROOT}":"${ROOT}" \
   -e ROOT="${ROOT}" \
   --name aaecaptcha "${AAECAPTCHA_IMG}" \
   bash -c "cd ${ROOT} && ./utils/extract_process_librispeech.sh"
