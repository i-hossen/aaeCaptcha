#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

TEST_FILE="${DATA_ROOT}/test_pgd_${EPSILON}_${STEPS}_${ALPHA}.csv"
PRED_FILE="${LOG}/preds_lingvo.txt"
INFER_SCRIPT="transcribe_lingvo.py"

cd "$ROOT"
check_project_root

docker run --gpus all --rm -it \
    -v "${ROOT}":${ROOT} \
    -e ROOT="${ROOT}" \
    -e LINGVO_ROOT="${LINGVO_ROOT}" \
    -e TEST_FILE="${TEST_FILE}" \
    -e PRED_FILE="${PRED_FILE}" \
    -e INFER_SCRIPT="${INFER_SCRIPT}" \
    --name lingvo "${LINGVO_IMG}" \
    bash -c "cd ${LINGVO_ROOT} && python ${INFER_SCRIPT} --input ${TEST_FILE} --output ${PRED_FILE}"
