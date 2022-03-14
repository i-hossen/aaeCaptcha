#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

INFER_SCRIPT="transcribe_kaldi.sh"
INFER_PARAM="adv"

cd "${ROOT}"
check_project_root

docker run --gpus all --rm -it \
    -v "${ROOT}":"${ROOT}" \
    -v ${KALDI_ROOT}/data:/opt/kaldi/egs/librispeech/s5/data \
    -v ${KALDI_ROOT}/exp:/opt/kaldi/egs/librispeech/s5/exp \
    -e ROOT="${ROOT}" \
    -e KALDI_ROOT="${KALDI_ROOT}" \
    -e INFER_SCRIPT="${INFER_SCRIPT}" \
    -e INFER_PARAM="${INFER_PARAM}" \
    --name kaldi_asr "${KALDI_IMG}" \
    bash -c "cd /opt/kaldi/egs/librispeech/s5 && cp ${KALDI_ROOT}/${INFER_SCRIPT} . && bash ${INFER_SCRIPT} ${INFER_PARAM}"
    