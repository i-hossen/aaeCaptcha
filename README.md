# aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA
This Github repo contains the code for the IEEE EuroS&P 2022 paper ["aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA"](https://arxiv.org/abs/2203.02735). Please refer to the [paper](https://arxiv.org/abs/2203.02735) for the details of the implementation.

# Setting up the base environment
The aaeCAPTCHA system generates audio adversarial CAPTCHAs against the DeepSpeech ASR model. First, you need to set up the base environment with DeepSpeech and download and preprocess the LibriSpeech dataset. We provided a Dockerfile to build a Docker image with all the required Python packages. We **highly recommend** using the **Docker installation** process to run the project's code. However, if you want to install the project directly, we also provide instructions for doing so. 

## Docker installation
```
git clone https://github.com/i-hossen/aaeCaptcha.git
cd aaeCaptcha
./setup_base.sh
```

This will set up the DeepSpeech environment, download and process the LibriSpeech dataset, and build the Docker image.

## Direct installation
```
git clone https://github.com/i-hossen/aaeCaptcha.git
cd aaeCaptcha

# Clone DeepSpeech repo
git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech && git checkout v0.4.1
cd ..

# Download and extract DeepSpeech v0.4.1 pretrained model
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz
tar -xvf deepspeech-0.4.1-checkpoint.tar.gz

# Install required python packages
pip install -r DeepSpeech/requirements.txt
pip uninstall tensorflow
pip install tensorflow-gpu==1.12 progressbar2 tqdm gdown

# Install CTC decoder 
cd DeepSpeech/native_client/ctcdecode && make bindings
cd ../../..
pip install DeepSpeech/native_client/ctcdecode/dist/*.whl

# Download and preprocess the LibriSpeech dataset
./utils/extract_process_librispeech.sh

# Install the FFmpeg system package required to run the code
sudo apt install -y ffmpeg
```

# Generating and evaluating audio adversarial CAPTCHAs

## Generate adversarial CAPTCHAs

```
python generate_aaeCaptcha.py --input data/test.csv --batch_size=25 --epsilon=350 --alpha=0 --steps=50  
```

The `--epsilon`, `--alpha`, and `--steps` options are hyperparameters for the PGD algorithm. The generated adversarial samples will be stored in the `data/audio_captchas` directory.

## Evaluate adversarial CAPTCHAs

1. Get the transcriptions predicted by the model:
```
python transcribe.py --test_files data/test_pgd_350_50_40.csv --test_batch_size 50 --checkpoint_dir deepspeech-0.4.1-checkpoint --pred_file logs/ds_out.txt
```
2. Compute attack metrics (WER, SRoA, etc.):
```
python compute_attack_perf_metrics.py --orig_path data/test.csv --pred_path logs/ds_out.txt
```

# Audio preprocessing attack against aaeCAPTCHA
We considered the following audio preprocessing and compression techniques in the paper:
1. Quantization (QT)
2. Down-sampling (DS)
3. Average Smoothing (AS)
4. Median Smoothing (MS)
5. Low-pass filtering (LPF)
6. Band-pass filtering (BPF)
7. MP3 compression
8. OPUS compression
9. AAC compression
10. SPEEX compression

Below, we give examples of using the code to perform audio preprocessing attacks on the generated adversarial samples and compute the attack performance of the primary model, DeepSpeech, on the preprocessed samples.

```
# run quantization attack with q=512
python attack/run_audio_preprocessing_attack.py --attack_type QT --param 512 --save

# compute the attack performance
./eval_attacks.sh -a QT -p 512

# run the band-pass filtering attack with lower_cutoff=100 and higher_cutoff=3000.
python attack/run_audio_preprocessing_attack.py --attack_type BPF --param 100 --param2 3000 --save

# compute the attack performance
./eval_attacks.sh -a BPF -p 100_3000

# run MP3 compression attack
python attack/run_audio_compression_attack.py --to mp3

# compute the attack performance
./eval_attacks.sh -a AC -p MP3
```

# Attack using various state-of-the-art ASR systems
We provided scripts to quickly configure and set up DeepSpeech 2, Jasper, Wave2Letter+, Lingvo, and Kaldi ASR models for automatically transcribing adversarial samples generated by the aaeCAPTCHA system. The installation scripts can be found in the `utils/` directory.

## Installing and setting up the attacker models
1. Set up DeepSpeech 2, Jasper, and Wave2Letter+ ASR models:
```
./utils/setup_ds2_jasper_w2lplus.sh
```
2. Set up Lingvo ASR:

```
./utils/setup_lingvo_asr.sh
```
3. Set up Kaldi ASR:
```
./utils/setup_kaldi_asr.sh
```

## Performing the attack 
Once the models are properly installed and configured for the inference, you can use the `transcribe_*` scripts in the `utils/` directory to get the models' predictions for the audio samples.

### Attack using DeepSpeech 2, Jasper, and Wave2Letter+.
```
# Get DeepSpeech 2, Jasper, or Wave2Letter+ model's predictions on the adversarial audios:
./utils/transcribe_ds2_jasper_w2lplus.sh

# Compute the attack performance.
python compute_attack_perf_metrics.py --orig_path data/test.csv --pred_path [PRED_FILE]
```

Replace "[PRED_FILE]" with the actual location of the predicted output file. The predictions, by default, will be saved in the `logs/` directory. You will need to modify the `INFER_PARAM` variable in the `transcribe_ds2_jasper_w2lplus.sh` script according to the model's name and the audio type. For example, to generate DeepSpeech 2's transcriptions for adversarial audios, set `INFERPARAM="ds2 adv"`.

### Attack using Lingvo 
```
# Get the Lingvo ASR model's predictions on adversarial audios
./utils/transcribe_lingvo.sh

# Compute the attack performance
python compute_attack_perf_metrics.py --orig_path data/test.csv --pred_path [PRED_FILE]
```

### Attack using Kaldi

```
# Get Kaldi's predictions on adversarial audios
./utils/transcribe_kaldi.sh

# Compute the attack performance
python compute_attack_perf_metrics.py --orig_path data/test.csv --pred_path [PRED_FILE]
```
