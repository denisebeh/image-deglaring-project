FROM --platform=linux/amd64 ubuntu:latest

WORKDIR /app
COPY . .

# Install core linux tools
RUN apt-get update
RUN apt-get install -y --no-install-recommends wget git ssh curl \
    software-properties-common build-essential unzip gcc \
    && rm -rf /var/lib/apt/lists/*

# Install opencv dependencies
RUN apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Decompress result dir
RUN unzip ./result.zip
RUN rm ./result.zip

# Decompress VGG Model dir
RUN unzip ./VGG_Model.zip
RUN rm ./VGG_Model.zip

# Install miniconda
RUN mkdir -p ./miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ./miniconda3/miniconda.sh
RUN bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ./miniconda3/bin/conda init bash

# Add conda to path
ENV PATH=/app/miniconda3/bin/:$PATH
ENV bashrc /root/.bashrc

# Install opencv
RUN conda create -n app python=3.8
RUN conda init && . $bashrc && conda activate app && conda install -y opencv

# Install python dependencies
RUN conda init && . $bashrc && conda activate app && python3 -m pip install --upgrade pip
RUN conda init && . $bashrc && conda activate app && python3 -m pip install setuptools
RUN conda init && . $bashrc && conda activate app && python3 -m pip install -r ./requirements.txt

# Run app
CMD conda init && . $bashrc && conda activate app && python3 ./launch.py
