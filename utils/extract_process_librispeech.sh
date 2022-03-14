#!/bin/bash

DATA_DIR=data/librispeech

if [ ! -d "${DATA_DIR}" ]; then
  mkdir "${DATA_DIR}" || exit 1
fi

cd DeepSpeech && python bin/import_librivox.py "../${DATA_DIR}"

if [ $? -ne 0 ]; then
  echo "failed to download and extract the LibriSpeech dataset." 1>&2
else
  echo "successfully downloaded and extracted the LibriSpeech dataset in ${DATA_DIR}"
fi