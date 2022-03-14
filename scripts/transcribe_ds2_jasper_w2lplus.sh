#!/bin/bash
EPSILON=350
STEPS=50
ALPHA=40
log="${ROOT}/logs"
test_file="${ROOT}/data/test.csv"
adv_file="${ROOT}/data/test_pgd_${EPSILON}_${STEPS}_${ALPHA}.csv"

usage() {
  echo "usage: $0 MODEL AUDIO_TYPE" 1>&2
  echo "usage: $0 [MODEL AUDIO_TYPE PARAM]" 1>&2
  echo "MODEL: ds2|jasper|w2lplus" 1>&2
  echo "AUDIO_TYPE: clean|adv|preprocessed" 1>&2
  echo "example: $0 ds2 clean|adv" 1>&2
  echo "example: $0 ds2 preprocessed QT_512" 1>&2
  exit 0
}

if [ ! -f "${test_file}" ] || [ ! -f ${adv_file} ]; then
  echo "test files ${test_file} or ${adv_file} do not exist."
  echo "aborting."
  exit 1
fi

case "$1" in
"ds2")
  OUT="${log}/ds2_out"
  config='example_configs/speech2text/ds2_large_8gpus_mp.py'
  logdir='ds2_large/'
  ;;
"jasper")
  OUT="${log}/jasper_out"
  config='example_configs/speech2text/jasper10x5_LibriSpeech_nvgrad_masks.py'
  logdir='jasper-10x5_dr_sp_novograd_masks/checkpoint/'
  ;;

"w2lplus")
  OUT="${log}/w2lplus_out"
  config='example_configs/speech2text/w2lplus_large_8gpus_mp.py'
  logdir='w2l_plus_large/'
  ;;
*)
  echo "invalid option: $1"
  echo "please choose one of the following: [ds2, jasper, w2lplus]"
  exit 1
  ;;
esac

case "$2" in
clean)
  cat "${test_file}" >/tmp/test.csv
  OUT="${OUT}_clean.txt"
  ;;
adv)
  cat "${adv_file}" >/tmp/test.csv
  OUT="${OUT}_pgd_${EPSILON}_${STEPS}_${ALPHA}.txt"
  ;;
preprocessed)
  if [ -z "$3" ]; then
    echo "a valid attack param is not provided."
    exit 1
  fi
  param="$3"
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

  cat "${adv_file}" >/tmp/test.csv
  sed -i "s#.wav,#_${param}.wav,#g" /tmp/test.csv
  OUT="${OUT}_pgd_${EPSILON}_${STEPS}_${ALPHA}_${param}.txt"
  ;;
*)
  echo "invalid argument: $2"
  usage
  ;;
esac

if [[ ! -d ${logdir} || ! -f $config ]]; then
  echo "some of the required files or dirs are missing."
  echo "aborting."
  exit 1
fi

if [[ "$1" =~ 'w2l' ]]; then
  python run.py --mode=infer --config_file $config --logdir $logdir --num_gpus=1 --use_horovod=False --infer_output_file $OUT
else
  python run.py --mode=infer --config_file $config --logdir $logdir --num_gpus=1 --use_horovod=False \
    --decoder_params/use_language_model=False --infer_output_file $OUT
fi

if [ $? -eq 0 ]; then
  echo "model's predictions have been written to ${OUT}"
fi