#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

chain_1d="https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz"
lm="https://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz"
ivector="https://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz"

cd "${ROOT}"
check_project_root

[ ! -d attack_ASR_models ] && (mkdir attack_ASR_models || exit 1)
cd attack_ASR_models || exit 1
mkdir kaldi_asr
cd kaldi_asr

# Download Kaldi Librispeech ASR model
for url in "${chain_1d}" "${lm}" "${ivector}"; do
    if ! wget "$url"; then
        echo "failed to download the file from "
        echo "$url."
        echo "please manually download the file and"
        echo "extract it to ${ATTACK_ASR_ROOT}/kaldi_asr"
        exit 1
    fi
done

for f in 0013_librispeech_v1_chain.tar.gz 0013_librispeech_v1_lm.tar.gz 0013_librispeech_v1_extractor.tar.gz; do
    tar -xvf "$f" || exit 1
    rm "$f"
done

# Setup Kaldi decoding env
# The following approach is a bit buggy. But it gets the job done!

[ ! -d data/test ] && (mkdir data/test || exit 1)
N_SAMPLES=$(wc -l ${DATA_ROOT}/test.csv | awk -F' ' '{print $1}')
N_SAMPLES=$((--N_SAMPLES))

for i in $(seq 1 1 ${N_SAMPLES}); do
    echo "librispeech-A_0000${i}"
done | sort >/tmp/utt.txt

awk -F',' 'FNR>1 {print $1}' ${DATA_ROOT}/test.csv >/tmp/wav.txt
paste -d ' ' /tmp/utt.txt /tmp/wav.txt >data/test/wav.scp.orig
sed 's/$/ librispeech-A/g' /tmp/utt.txt >data/test/utt2spk
awk -F ',' 'FNR>1 {print $NF}' ${DATA_ROOT}/test.csv | tr 'a-z' 'A-Z' >data/test/text
####

cd "${ROOT}"
check_project_root

cp scripts/transcribe_kaldi.sh "${KALDI_ROOT}"

docker pull kaldiasr/kaldi:gpu-latest || exit 1

# sudo chown -R "${USER}" "${KALDI_ROOT}"
rm /tmp/wav.txt /tmp/utt.txt
