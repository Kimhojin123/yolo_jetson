# yolo_jetson

```
dli@dli-desktop:~$ wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
```
```
dli@dli-desktop:~$ ls
결과
Archiconda3-0.2.3-Linux-aarch64.sh  jetson-fan-ctl  Templates
Desktop                             meno            USB-Camera
Documents                           Music           Videos
Downloads                           Pictures
examples.desktop                    Public
```
```
sudo chmod 755 Archiconda3-0.2.3-Linux-aarch64.sh 명령어는 파일에 대해 모든 사용자가 읽고 실행할 수 있도록 하면서, 소유자는 추가로 쓰기 권한도 부여하는 작업을 수행한다.
```
```
./Archiconda3-0.2.3-Linux-aarch64.sh 입력
```
enter입력
```
>>>yes
결과
Archiconda3 will now be installed into this location:
/home/dli/archiconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/dli/archiconda3] >>> enter입력
PREFIX=/home/dli/PREFIX=/home/ldh/archiconda3
installing: python-3.7.1-h39be038_1002 ...
Python 3.7.1
installing: ca-certificates-2018.03.07-0 ...
installing: conda-env-2.6.0-1 ...
installing: libgcc-ng-7.3.0-h5c90dd9_0 ...
installing: libstdcxx-ng-7.3.0-h5c90dd9_0 ...
installing: bzip2-1.0.6-h7b6447c_6 ...
installing: libffi-3.2.1-h71b71f5_5 ...
installing: ncurses-6.1-h71b71f5_0 ...
installing: openssl-1.1.1a-h14c3975_1000 ...
installing: xz-5.2.4-h7ce4240_4 ...
installing: yaml-0.1.7-h7ce4240_3 ...
installing: zlib-1.2.11-h7b6447c_2 ...
installing: readline-7.0-h7ce4240_5 ...
installing: tk-8.6.9-h84994c4_1000 ...
installing: sqlite-3.26.0-h1a3e907_1000 ...
installing: asn1crypto-0.24.0-py37_0 ...
installing: certifi-2018.10.15-py37_0 ...
installing: chardet-3.0.4-py37_1 ...
installing: idna-2.7-py37_0 ...
installing: pycosat-0.6.3-py37h7b6447c_0 ...
installing: pycparser-2.19-py37_0 ...
installing: pysocks-1.6.8-py37_0 ...
installing: ruamel_yaml-0.15.64-py37h7b6447c_0 ...
installing: six-1.11.0-py37_1 ...
installing: cffi-1.11.5-py37hc365091_1 ...
installing: setuptools-40.4.3-py37_0 ...
installing: cryptography-2.5-py37h9d9f1b6_1 ...
installing: wheel-0.32.1-py37_0 ...
installing: pip-10.0.1-py37_0 ...
installing: pyopenssl-18.0.0-py37_0 ...
installing: urllib3-1.23-py37_0 ...
installing: requests-2.19.1-py37_0 ...
installing: conda-4.5.12-py37_0 ...
installation finished.
```
```
conda env list 입력
결과 # conda environments:
#
base                  *  /home/dli/yes
conda activate base
jetson_release 입력
결과
Software part of jetson-stats 4.2.8 - (c) 2024, Raffaello Bonghi
Jetpack missing!
 - Model: NVIDIA Jetson Nano Developer Kit
 - L4T: 32.7.5
NV Power Mode[0]: MAXN
Serial Number: [XXX Show with: jetson_release -s XXX]
Hardware:
 - P-Number: p3448-0000
 - Module: NVIDIA Jetson Nano (4 GB ram)
Platform:
 - Distribution: Ubuntu 18.04 Bionic Beaver
 - Release: 4.9.337-tegra
jtop:
 - Version: 4.2.8
 - Service: Active
Libraries:
 - CUDA: 10.2.300
 - cuDNN: 8.2.1.32
 - TensorRT: 8.2.1.8
 - VPI: 1.2.3
 - Vulkan: 1.2.70
 - OpenCV: 4.1.1 - with CUDA: NO
```
phython3.8 가상환경을 만든다.

```
conda deactivate 입력
결과
(base) dli@dli-desktop:~$ 앞에 (base가 사라짐.
```
```
conda create -n yolo python=3.8 -y 입력
결과
새로운 패키지 다운로드
```
```
 conda activate yolo 입력
결과 
(yolo) dli@dli-desktop:~$
```
 pip install -U pip wheel gdown
