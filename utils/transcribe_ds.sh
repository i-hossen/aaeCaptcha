#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

DS_CHKPOINT="deepspeech-0.4.1-checkpoint"

INFER_SCRIPT="transcribe.py"
TEST_FILE="${DATA_ROOT}/test_pgd_${EPSILON}_${STEPS}_${ALPHA}.csv"
PRED_FILE="$LOG/ds_out.txt"

cd "${ROOT}"
check_project_root

docker run --gpus all --rm -it \
    -v "${ROOT}":"${ROOT}" \
    -e ROOT="${ROOT}" \
    -e INFER_SCRIPT="${INFER_SCRIPT}" \
    -e TEST_FILE="${TEST_FILE}" \
    -e PRED_FILE="${PRED_FILE}" \
    --name aaecaptcha "${AAECAPTCHA_IMG}" \
    bash -c "cd ${ROOT} && python ${INFER_SCRIPT} --test_files ${TEST_FILE} --pred_file ${PRED_FILE} --test_batch_size 50 --checkpoint_dir ${DS_CHKPOINT}"
