#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

LINGVO_DEVICE="gpu"
LINGVO_DIR=/tmp/lingvo
LINGVO_COMPILED="${LINGVO_ROOT}/lingvo_compiled"

git -C /tmp clone https://github.com/yaq007/lingvo.git
cd /tmp/lingvo || exit 1
git checkout icml

cd "${ROOT}"
check_project_root

[ ! -d attack_ASR_models ] && (mkdir attack_ASR_models || exit 1)
cd attack_ASR_models || exit 1
mkdir lingvo_asr
cd lingvo_asr
mkdir lingvo_compiled

docker build --no-cache --tag "${LINGVO_IMG}" - <"${ROOT}/docker/lingvo_asr.dockerfile" || exit 1

docker run --rm $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo \
    -v ${LINGVO_COMPILED}:/tmp/lingvo_compiled \
    --name lingvo tensorflow:lingvo bash -c 'bazel build -c opt --config=cuda //lingvo:trainer && cp -rfL bazel-bin/lingvo/trainer.runfiles/__main__/lingvo /tmp/lingvo_compiled'

if [[ $? -ne 0 ]]; then
    echo "failed to compile lingvo."
    exit 1
fi

cp -rfL "${LINGVO_COMPILED}/lingvo" .
# sudo chown -R $USER ${LINGVO_COMPILED}
# PYTHONPATH=$PYTHONPATH:$LINGVO_COMPILED

cd "${LINGVO_ROOT}" || exit 1
mkdir model
cd model
wget https://raw.githubusercontent.com/cleverhans-lab/cleverhans/v3.1.0/examples/adversarial_asr/model/checkpoint
wget https://github.com/cleverhans-lab/cleverhans/raw/v3.1.0/examples/adversarial_asr/model/ckpt-00908156.index
wget https://github.com/cleverhans-lab/cleverhans/raw/v3.1.0/examples/adversarial_asr/model/ckpt-00908156.meta
wget http://cseweb.ucsd.edu/~yaq007/ckpt-00908156.data-00000-of-00001

for f in checkpoint ckpt-00908156.index ckpt-00908156.meta ckpt-00908156.data-00000-of-00001; do
    if [ ! -f "$f" ]; then
        echo "failed to download $f"
        echo "please manually download the file and save it in "
        echo "${KALDI_ROOT}/model directory."
        exit 1
    fi
done

cd "${ROOT}"
check_project_root

cp scripts/transcribe_lingvo.py "${LINGVO_ROOT}"

rm -rf /tmp/lingvo
