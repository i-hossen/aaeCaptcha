FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN alias pip='pip3'
RUN alias python='python3'
RUN alias ipython='ipython3'
RUN alias ..='cd ..'

RUN chmod 1777 /tmp

RUN apt-get update -y
RUN apt-get install -y locales locales-all language-pack-en

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

RUN apt-get install -y \
 swig \
 sox \
 libsox-dev \
 python-pyaudio \
 git \
 wget \
 python-pip \
 python-dev \
 silversearcher-ag \
 ranger \
 ffmpeg \
 python3-levenshtein \
 libffi6 libffi-dev \
 llvm-8 

ENV LLVM_CONFIG=/usr/bin/llvm-config-8

RUN pip3 install \
absl-py==0.15.0 \
astor==0.8.1 \
attrdict==2.0.1 \
attrs==21.4.0 \
audioread==2.1.9 \
backcall==0.2.0 \
beautifulsoup4==4.10.0 \
bs4==0.0.1 \
certifi==2018.8.24 \
cffi==1.15.0 \
chardet==4.0.0 \
cryptography==3.2.1 \
decorator==5.1.1 \
defusedxml==0.7.1 \
entrypoints==0.3 \
future==0.18.2 \
gast==0.5.3 \
grpcio==1.41.1 \
h5py==2.10.0 \
idna==2.10 \
importlib-metadata==2.1.2 \
importlib-resources==3.2.1 \
ipython==7.9.0 \
ipython-genutils==0.2.0 \
jedi==0.17.2 \
Jinja2==2.11.3 \
joblib==0.14.1 \
jsonschema==3.2.0 \
jupyter-client==6.1.12 \
jupyter-core==4.6.3 \
Keras-Applications==1.0.8 \
Keras-Preprocessing==1.1.2 \
librosa==0.7.2 \
llvmlite==0.32.1 \
Markdown==3.2.2 \
MarkupSafe==1.1.1 \
mistune==0.8.4 \
mock==3.0.5 \
nbformat==5.1.3 \
numba==0.47.0 \
numexpr==2.8.1 \
numpy==1.18.5 \
packaging==20.9 \
pandas==0.24.0 \
pandocfilters==1.5.0 \
parso==0.7.1 \
pexpect==4.8.0 \
pickleshare==0.7.5 \
progressbar==2.5 \
progressbar2==3.55.0 \
prompt-toolkit==2.0.10 \
protobuf==3.19.3 \
ptyprocess==0.7.0 \
pybind11==2.9.0 \
pydub==0.25.1 \
Pygments==2.11.2 \
pyparsing==2.4.7 \
pyrsistent==0.17.3 \
python-dateutil==2.8.2 \
python-speech-features==0.6 \
python-utils==2.7.1 \
pytz==2021.3 \
pyxdg==0.27 \
pyzmq==20.0.0 \
requests==2.25.1 \
resampy==0.2.2 \
scikit-learn==0.22.2.post1 \
scipy==1.4.1 \
six==1.16.0 \
SoundFile==0.10.3.post1 \
soupsieve==2.1 \
sox==1.4.1 \
soxr==0.1.1 \
tables==3.5.2 \
tensorboard==1.12.2 \
tensorflow-gpu==1.12.0 \
termcolor==1.1.0 \
testpath==0.5.0 \
torch==1.5.1 \
tornado==6.1 \
tqdm==4.63.0 \
traitlets==4.3.3 \
urllib3==1.26.8 \
wcwidth==0.2.5 \
webencodings==0.5.1 \
Werkzeug==1.0.1 \
zipp==1.2.0 

RUN git clone -b v0.4.1 https://github.com/mozilla/DeepSpeech.git

RUN wget https://github.com/git-lfs/git-lfs/releases/download/v2.8.0/git-lfs-linux-amd64-v2.8.0.tar.gz
RUN tar -xvzf git-lfs-linux-amd64-v2.8.0.tar.gz
RUN ./install.sh
RUN git lfs install
RUN git lfs --version

RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

RUN cd DeepSpeech/native_client/ctcdecode && make bindings NUM_PROCESSES=8
RUN pip3 install DeepSpeech/native_client/ctcdecode/dist/*.whl

# ENTRYPOINT /bin/bash
CMD ["/bin/bash"]
