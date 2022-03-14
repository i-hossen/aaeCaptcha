#!/bin/bash
. ./paths.sh || exit 1
. ./util.sh || exit 1

ds2_url="https://drive.google.com/uc?id=1EDvL9wMCO2vVE-ynBvpwkFTultbzLNQX"
jasper_url="https://drive.google.com/uc?id=12CQvNrTvf0cjTsKjbaWWvdaZb7RxWI6X"
w2lplus_url="https://drive.google.com/uc?id=10EYe040qVW6cfygSZz6HwGQDylahQNSa"

manual_installation() {
  echo "failed to download and extract pretrained models."
  echo "please manually download and extract the models in"
  echo "attack_ASR_models/ds2_jasper_w2lplus/OpenSeq2Seq directory."
  echo "refer to https://nvidia.github.io/OpenSeq2Seq/html/installation.html#installation-of-openseq2seq-for-speech-recognition"
  echo "for installation instruction details."
  exit 1
}

cd "$ROOT"
check_project_root

[ ! -d attack_ASR_models ] && (mkdir attack_ASR_models || exit 1)
cd attack_ASR_models || exit 1
mkdir ds2_jasper_w2lplus
cd ds2_jasper_w2lplus
git clone https://github.com/NVIDIA/OpenSeq2Seq.git
cd OpenSeq2Seq || exit 1

if ! which gdown >/dev/null; then
  echo "couldn't find gdown in the system path."
  echo "this program requires the gdown script to download files from Google Drive."
  echo "cannot proceed."
  exit 1
fi

# Download DeepSpeech 2 pretrained model
if [ ! -f "ds2_large.tar.gz" ]; then
  gdown -O "ds2_large.tar.gz" "${ds2_url}" || manual_installation
fi
# Download Jasper pretrained model
if [ ! -f "jasper-10x5_dr_sp_novograd_masks.tar.gz" ]; then
  gdown -O "jasper-10x5_dr_sp_novograd_masks.tar.gz" "${jasper_url}" || manual_installation
fi
# Download Wave2Letter+ pretrained model
if [ ! -f "w2l_plus_large.tar.gz" ]; then
  gdown -O "w2l_plus_large.tar.gz" "${w2lplus_url}" || manual_installation
fi

for f in ds2_large.tar.gz w2l_plus_large.tar.gz jasper-10x5_dr_sp_novograd_masks.tar.gz; do
  if [ ! -f "$f" ]; then
    echo "the file $f has not been fetched successfully."
    echo "cannot proceed."
    exit 1
  fi
done

if ! [[ $(md5sum ds2_large.tar.gz | awk '{print $1}') = "c5cbee2e82635406719cfd2f33fa3b1f" ]]; then
  echo "checksum validation failed for ds2_large.tar.gz"
  echo "please manually download the file from "
  echo "${ds2_url}"
  rm ds2_large.tar.gz
  exit 1
fi

if ! [[ $(md5sum jasper-10x5_dr_sp_novograd_masks.tar.gz | awk '{print $1}') = "8c61365564dedcc0cdb01514bf356b79" ]]; then
  echo "checksum validation failed for jasper-10x5_dr_sp_novograd_masks.tar.gz"
  echo "please manually download the file from "
  echo "${jasper_url}"
  rm jasper-10x5_dr_sp_novograd_masks.tar.gz
  exit 1
fi

if ! [[ $(md5sum w2l_plus_large.tar.gz | awk '{print $1}') = "8d50c5d5d87ecec122c31ace47cf8e9c" ]]; then
  echo "checksum validation failed for w2l_plus_large.tar.gz"
  echo "please manually download the file from"
  echo "${w2lplus_url}"
  rm w2l_plus_large.tar.gz
  exit 1
fi

for f in ds2_large.tar.gz w2l_plus_large.tar.gz; do
  tar -xvf "$f" || exit 1
  rm "$f"
done

mkdir jasper-10x5_dr_sp_novograd_masks
tar -xvf jasper-10x5_dr_sp_novograd_masks.tar.gz -C jasper-10x5_dr_sp_novograd_masks || exit 1
rm jasper-10x5_dr_sp_novograd_masks.tar.gz

for f in example_configs/speech2text/ds2_large_8gpus_mp.py example_configs/speech2text/jasper10x5_LibriSpeech_nvgrad_masks.py example_configs/speech2text/w2lplus_large_8gpus_mp.py; do
  if [ ! -f "$f" ]; then
    echo "config file $f doesn't exist."
    exit 1
  fi
done

{
  echo 'infer_params = {'
  echo '    "data_layer": Speech2TextDataLayer,'
  echo '    "data_layer_params": {'
  echo '        "num_audio_features": 160,'
  echo '        "input_type": "spectrogram",'
  echo '        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",'
  echo '        "dataset_files": ['
  echo '            "/tmp/test.csv",'
  echo '        ],'
  echo '        "shuffle": False,'
  echo '    },'
  echo '}'
} >>example_configs/speech2text/ds2_large_8gpus_mp.py

{
  echo 'infer_params = {'
  echo '    "data_layer": Speech2TextDataLayer,'
  echo '    "data_layer_params": {'
  echo '        "dataset_files": ['
  echo '            "/tmp/test.csv",'
  echo '        ],'
  echo '        "shuffle": False,'
  echo '    },'
  echo '}'
} >>example_configs/speech2text/jasper10x5_LibriSpeech_nvgrad_masks.py

{
  echo 'infer_params = {'
  echo '    "data_layer": Speech2TextDataLayer,'
  echo '    "data_layer_params": {'
  echo '        "num_audio_features": 64,'
  echo '        "input_type": "logfbank",'
  echo '        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",'
  echo '        "dataset_files": ['
  echo '            "/tmp/test.csv",'
  echo '        ],'
  echo '        "shuffle": False,'
  echo '    },'
  echo '}'
} >>example_configs/speech2text/w2lplus_large_8gpus_mp.py

cd "$ROOT"
check_project_root
cp scripts/transcribe_ds2_jasper_w2lplus.sh "${ATTACK_ASR_ROOT}"/ds2_jasper_w2lplus/OpenSeq2Seq

docker pull "${OPENSEQ2SEQ_IMG}" || exit 1
