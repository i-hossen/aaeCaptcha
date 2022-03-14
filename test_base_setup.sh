#!/bin/bash
TESTDIR="/tmp/aaeCaptcha"

rm_test_dir() {
    if [ -d "${TESTDIR}" ]; then
        rm -rf "${TESTDIR}" || exit 1
    fi
}
cd ~
rm_test_dir

git -C /tmp clone https://github.com/i-hossen/aaeCaptcha.git || exit 1

cd "${TESTDIR}" || exit 1

./setup_base.sh

. ./paths.sh || exit 1
. ./util.sh || exit 1

docker run --gpus all --rm -it \
-v "${ROOT}":"${ROOT}" \
-w "${ROOT}" \
-e ROOT="${ROOT}" \
--name aaecaptcha \
"${AAECAPTCHA_IMG}" \
bash -c '
python generate_aaeCaptcha.py --input data/test.csv --epsilon 350 --steps 50 --alpha 40 &&\
python transcribe.py --test_files data/test_pgd_350_50_40.csv --test_batch_size 50 --pred_file logs/ds_out.txt --checkpoint_dir deepspeech-0.4.1-checkpoint &&\
python compute_attack_perf_metrics.py --orig_path data/test.csv --pred_path logs/ds_out.txt
'
cd ~
rm_test_dir
