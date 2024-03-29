FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gcc-4.8 g++-4.8 gcc-4.8-base \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python \
        python-dev \
        python-tk \
        rsync \
        software-properties-common \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100 &&\
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100

RUN curl -O https://bootstrap.pypa.io/pip/2.7/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        jupyter_http_over_ws \
        matplotlib \
        numpy \
        pandas \
        recommonmark \
        scipy \
        sklearn \
        sphinx \
        sphinx_rtd_theme \
        && \
    python -m ipykernel.kernelspec

# RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN pip --no-cache-dir install tensorflow-gpu==1.13.1 grpcio==1.39.0 librosa==0.7.2 cffi==1.14.6 llvmlite==0.31.0

ARG bazel_version=0.17.2
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


# TensorBoard
# EXPOSE 6006

# Jupyter
# EXPOSE 8888

WORKDIR "/tmp/lingvo"

CMD ["/bin/bash"]
