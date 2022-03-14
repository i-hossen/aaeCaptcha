#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

test_orig_file=${ROOT}/data/test.csv
test_adv_file=${ROOT}/data/test_pgd_${EPSILON}_${STEPS}_${ALPHA}.csv

epsilon=${EPSILON}
steps=${STEPS}
alpha=${ALPHA}

usage() {
    echo "usage: $0 [ -a ATTACK_TYPE ] [ -p ATTACK_PARAM ]" 1>&2
    echo "example: $0 -a QT -p 512" 1>&2
    echo "example: $0 -a BPF -p 100_2000" 1>&2
    echo "example: $0 -a AC -p MP3" 1>&2
    exit 0
}

while getopts ":h:a:p:" o; do
    case "${o}" in
    a)
        attack=${OPTARG}
        ;;
    p)
        param=${OPTARG}
        ;;
    \?)
        echo "invalid option ${OPTARG}"
        usage
        ;;
    h | *)
        usage
        ;;
    esac
done

shift $((OPTIND - 1))

if [ -z "${attack}" ] || [ -z "${param}" ]; then
    usage
fi

if ! echo "${attack}" | egrep "^(QT|DS|AS|MS|LPF|BPF|AC)$" >/dev/null; then
    echo "invalid attack type: ${attack}" 1>&2
    echo "valid options: [QT, DS, AS, MS, LPF, BPF, AC]" 1>&2
    exit 1
fi

if [ "$attack" = "AC" ]; then
    if ! echo "${param}" | egrep "^(MP3|OPUS|AAC|SPX)$" >/dev/null; then
        echo "unsupported Audio Compression (AC) type ${param} provided." 1>&2
        echo "valid AC params: MP3|OPUS|AAC|SPX"
        exit 1
    fi
fi

if [ ! -f "${test_adv_file}" ] || [ ! -f "${test_orig_file}" ]; then
    echo "the files ${test_adv_file} or ${test_orig_file} do not exist" 1>&2
    echo "terminating." 1>&2
    exit 1
fi

cd "${ROOT}"
check_project_root

test_file_new="/tmp/test_pgd_${epsilon}_${steps}_${alpha}_${attack}_${param}.csv"
cat ${test_adv_file} >${test_file_new}
sed -i "s#.wav,#_${attack}_${param}.wav,#g" "${test_file_new}"
pred_file=/tmp/"ds_out_pgd_${epsilon}_${steps}_${alpha}_${attack}_${param}.txt"

checkpoint_dir='deepspeech-0.4.1-checkpoint'
test_batch_size=50

if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "DeepSpeech checkpoint directory ${checkpoint_dir} does not exist." 1>&2
    echo "terminating." 1>&2
    exit 1
fi

python transcribe.py --test_files "${test_file_new}" --test_batch_size "${test_batch_size}" --checkpoint_dir "${checkpoint_dir}" --pred_file "${pred_file}"

if [[ $? -ne 0 ]]; then
    echo "" 1>&2
    echo "something unexpected occurred. aborting." 1>&2
    exit 1
fi

echo ""
echo ""

python compute_attack_perf_metrics.py --orig_path "${test_orig_file}" --pred_path "${pred_file}"

if [[ $? -ne 0 ]]; then
    echo ""
    echo "couldn't compute the attack performance."
    echo "check the program output above for errors."
    exit 1
fi

rm "${test_file_new}" "${pred_file}"
