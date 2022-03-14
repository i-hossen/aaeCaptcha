#!/bin/bash
epsilon=350
steps=50
alpha=40

log_dir="${ROOT}/logs"
test_orig="${ROOT}/data/test.csv"
test_adv="${ROOT}/data/test_pgd_${epsilon}_${steps}_${alpha}.csv"

usage() {
  echo "usage: $0 [AUDIO_TYPE]" 1>&2
  echo "usage: $0 [AUDIO_TYPE PARAM]" 1>&2
  echo "valid AUDIO_TYPE: [clean, adv, preprocessed]" 1>&2
  echo "example: $0 clean|adv" 1>&2
  echo "example: $0 preprocessed QT_512" 1>&2
  echo "example: $0 preprocessed AC_MP3" 1>&2
  exit 0
}

if [ ! -f "${test_orig}" ] || [ ! -f "${test_adv}" ]; then
  echo "test files ${test_orig} or ${test_adv} do not exist."
  echo "aborting."
  exit 1
fi

if [ ! -d "${log_dir}" ]; then
  echo "log dir ${log_dir} doesn't exist."
  echo "aborting."
  exit 1
fi

case $1 in
clean)
  test_file=${test_orig}
  pred_file="${log_dir}/preds_kaldi_clean.txt"
  awk -F',' 'FNR > 1 {print $1}' ${test_file} >"${pred_file}"
  awk -F' ' '{print $1}' data/test/wav.scp.orig >/tmp/dummy2.txt
  paste -d ' ' /tmp/dummy2.txt "${pred_file}" >data/test/wav.scp
  ;;
adv)
  test_file=${test_adv}
  pred_file="${log_dir}/preds_kaldi_pgd_${epsilon}_${steps}_${alpha}.txt"
  awk -F',' 'FNR > 1 {print $1}' ${test_file} >"${pred_file}"
  awk -F' ' '{print $1}' data/test/wav.scp.orig >/tmp/dummy2.txt
  paste -d ' ' /tmp/dummy2.txt "${pred_file}" >data/test/wav.scp
  ;;
preprocessed)
  if [ -z "$2" ]; then
    echo "a valid attack param is not provided" 1>&2
    exit 1
  fi
  param="$2"
  if ! echo "${param}" | egrep "^(QT|DS|AS|MS|LPF|BPF|AC)_" >/dev/null; then
    echo "invalid attack type/param: ${param}"
    echo "valid types: [QT, DS, AS, MS, LPF, BPF, AC]"
    exit 1
  fi

  if echo "${param}" | egrep "^AC_" >/dev/null; then
    if ! echo ${param} | egrep "(MP3|OPUS|AAC|SPX)$" >/dev/null; then
      echo "unsupported audio comp. (AC) type: ${param}."
      echo "valid params: [AC_MP3, AC_OPUS, AC_AAC, AC_SPX]"
      exit 1
    fi
  fi

  test_file="/tmp/test_pgd_${epsilon}_${steps}_${alpha}_${param}.csv"
  pred_file="${log_dir}/preds_kaldi_pgd_${epsilon}_${steps}_${alpha}_${param}.txt"
  awk -F',' 'FNR > 1 {print $1}' "${test_adv}" >"${pred_file}"
  sed -i "s#.wav#_${param}.wav#g" "${pred_file}"
  awk -F' ' '{print $1}' data/test/wav.scp.orig >/tmp/dummy2.txt
  paste -d ' ' /tmp/dummy2.txt "${pred_file}" >data/test/wav.scp
  ;;
*)
  echo "invalid argument: $1"
  usage
  ;;

esac

if ! [[ $(pwd) =~ 'librispeech/s5' ]]; then
  echo "you should run this script from /opt/kaldi/egs/librispeech/s5"
  echo "aborting."
  exit 1
fi

dataset=test

if [ ! -d "data/${dataset}" ]; then
  echo "the directory data/${dataset} doesn't exist."
  echo "make sure that the Kaldi ASR env is properly configured."
  echo ""
  exit 1
fi

# utils/fix_data_dir.sh data/${dataset}
# utils/validate_data_dir.sh data/${dataset}

utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk >data/${dataset}/spk2utt
utils/copy_data_dir.sh data/${dataset} data/${dataset}_hires

export train_cmd="run.pl"
export decode_cmd="run.pl --mem 2G"

steps/make_mfcc.sh \
  --nj 1 \
  --mfcc-config conf/mfcc_hires.conf \
  --cmd "$train_cmd" data/${dataset}_hires

steps/compute_cmvn_stats.sh data/${dataset}_hires
utils/fix_data_dir.sh data/${dataset}_hires

nspk=$(wc -l <data/${dataset}_hires/spk2utt)

steps/online/nnet2/extract_ivectors_online.sh \
  --cmd "$train_cmd" --nj "${nspk}" \
  data/${dataset}_hires exp/nnet3_cleaned/extractor \
  exp/nnet3_cleaned/ivectors_${dataset}_hires

export dir=exp/chain_cleaned/tdnn_1d_sp
# export dir=exp/chain_cleaned/tdnn_1d_AT_fgsm_sp
# export dir=exp/chain_cleaned/tdnn_1d_AT_pgd_sp

export graph_dir=$dir/graph_tgsmall

utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov \
  data/lang_${dataset}_tgsmall $dir $graph_dir

steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
  --nj 1 --cmd "$decode_cmd" \
  --online-ivector-dir exp/nnet3_cleaned/ivectors_${dataset}_hires \
  $graph_dir data/${dataset}_hires $dir/decode_${dataset}_tgsmall

# steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_${dataset}_{tgsmall,tglarge} \
#     data/${dataset}_hires $dir/decode_${dataset}_{tgsmall,tglarge}

# steps/get_ctm.sh data/${dataset} $dir/graph_tgsmall \
#     $dir/decode_${dataset}_tglarge

cat $dir/decode_${dataset}_tgsmall/log/decode.1.log | grep '^librispeech' | cut -f2- -d ' ' | tr 'A-Z' 'a-z' >/tmp/preds.txt
paste -d ',' "${pred_file}" /tmp/preds.txt >/tmp/dump.txt

# echo "writing decoded transcriptions to: ${pred_file}"

echo "wav_filename,transcript" >"${pred_file}"
cat /tmp/dump.txt >>"${pred_file}"

echo "model's predictions have been written to ${pred_file}"

rm /tmp/dummy2.txt /tmp/dump.txt /tmp/preds.txt
rm -rf "data/${dataset}/.backup"
rm -rf "data/${dataset}_hires"
