# Image De-glaring Project

## 1. Introduction

This project aims to train a deep learning image enhancement model that is able to remove the glares on images and deploys the solution as a web service. The web service exposes API endpoints to remove glare from input images. 

## 2. Install and build using Docker

1. Pull image-deglaring-project repository

``` bash
get clone https://github.com/denisebeh/image-deglaring-project.git
```

2. Install Docker Engine. Refer to https://docs.docker.com/engine/install/ to install Docker Engine.

3. Download VGG19 Model weights

``` bash
wget -O VGG_Model.zip "https://hkustconnect-my.sharepoint.com/:u:/g/personal/cleiaa_connect_ust_hk/EZeGsvuqh1dJr0E2Fxf6IKoBQ7wZpGi3NFqZxhzC8-3GHg?e=LLWUZT&download=1"
```

4. Download model checkpoints

``` bash
pip3 install gdown
gdown https://drive.google.com/file/d/1H98wT-dOloDICgQaRAoCEwGmv0FZ1VlL/view?usp=drive_link
```

5. Build our Docker image using Docker.

``` bash
docker build -f ./docker/app.dockerfile -t deglare-app .
```

## 3. Install and deploy locally with Anaconda

1. Pull image-deglaring-project repository

``` bash
get clone https://github.com/denisebeh/image-deglaring-project.git
```

2. Download VGG19 Model weights

``` bash
wget -O VGG_Model.zip "https://hkustconnect-my.sharepoint.com/:u:/g/personal/cleiaa_connect_ust_hk/EZeGsvuqh1dJr0E2Fxf6IKoBQ7wZpGi3NFqZxhzC8-3GHg?e=LLWUZT&download=1"
```

3. Download model checkpoints

``` bash
wget -O result.zip "https://www.dropbox.com/scl/fi/26p8e04i0vf4wn3mf9zad/result.zip?rlkey=3zx74ym8m1prejdee2rcdi6kj&dl=0"
```

4. Install Anaconda. Refer to https://docs.anaconda.com/free/anaconda/install/index.html to install.

5. Create and activate conda environment

``` bash
conda create -n image-deglare-proj python=3.8
conda activate image-deglare-proj
```

6. Install OpenCV binaries

``` bash
conda install -y opencv
``` 

7. Install the application dependencies

``` bash
python3 -m pip install -r ./requirements.txt
```

8. Launch web application

``` bash
python3 ./launch.py
```
