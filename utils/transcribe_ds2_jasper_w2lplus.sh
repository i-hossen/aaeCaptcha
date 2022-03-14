#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

INFER_SCRIPT="transcribe_ds2_jasper_w2lplus.sh"
INFER_PARAM="w2lplus adv"

cd "${ROOT}"
check_project_root

docker run --gpus all --rm -it \
    -v "${ROOT}":"${ROOT}" \
    -e ROOT="${ROOT}" \
    -e DS2_JASPER_W2LPLUS_ROOT="${DS2_JASPER_W2LPLUS_ROOT}" \
    -e INFER_SCRIPT="${INFER_SCRIPT}" \
    -e INFER_PARAM="${INFER_PARAM}" \
    --name openseq2seq_asrs "${OPENSEQ2SEQ_IMG}" \
    bash -c "cd ${DS2_JASPER_W2LPLUS_ROOT} && bash ${INFER_SCRIPT} ${INFER_PARAM}"
