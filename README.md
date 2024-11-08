##  yolo_jetson

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
```
 pip install -U pip wheel gdown
 gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
 gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV 입력

```
```

sudo apt-get install libopenblas-base libopenmpi-dev
sudo apt-get install libomp-dev
pip install torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl
pip install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
python -c "import torch; print(torch.__version__)"

```
```
 conda install numpy                                       # 또는 >>> 다음에 설치를 해도 된다.

```
```



(yolo) dli@dli:~$ python

>>> import torch
>>> import torchvision
>>> print(torch.__version__)
>>> print(torchvision.__version__)
>>> print("cuda used", torch.cuda.is_available())
cuda used True
>>>

```
```
git clone https://github.com/Tory-Hwang/Jetson-Nano2

```
```

(yolo) dli@dli:~$ cd Jetson-Nano2/
(yolo) dli@dli:~/Jetson-Nano2$ cd V8
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install ultralytics
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install -r requirements.txt 
(yolo) dli@jdli:~/Jetson-Nano2/V8$ pip install ffmpeg-python
(yolo) dli@dli:~/Jetson-Nano2$ sudo apt install tree
(yolo) dli@jdli:~/Jetson-Nano2$ tree -L 2


```
```


gedit  detectY8.py

```
```

(yolo) dli@jetson:~/Jetson-Nano2/V8$ python detectY8.py

```
```

def predict(cfg=DEFAULT_CFG, use_python=False): brtsp = True -> brtsp = False detectY8.py 스크립트에서 RTSP(Remote Desktop Protocol)를 사용하지 않도록 변경하려면, brtsp = True 라인을 찾아 False로 변경해야 함
이 변경은 스크립트가 RTSP 스트리밍 대신 다른 비디오 입력 소스(예: USB 카메라)를 사용하도록 지시함
코드에서 brtsp 변수가 RTSP 스트리밍을 활성화하거나 비활성화하는 데 사용됨

brtsp(Better RTSP) 설정의 목적:더 효율적인 RTSP 스트림 처리 버퍼링 최적화 프레임 스킵 방지 메모리 사용량 최적화

True로 설정할 때의 장점:더 안정적인 스트림 처리 지연시간 감소 메모리 효율성 향상

False로 변경하는 주요 이유들:하드웨어 리소스가 제한적일 때 실시간성보다 처리의 정확성이 더 중요할 때 메모리 사용량을 줄여야 할 때 레거시 시스템과의 호환성이 필요할 때

```
```


(yolo) yolo@yolo-desktop:~/Jetson-Nano2/V8$ python detectY8.py

```

```
# Ultralytics YOLO , GPL-3.0 license

import argparse
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
   
    brtsp = False
    if brtsp: 
        cfg.source = 'rtsp://admin:satech1234@192.168.0.151:554/udp/av0_0'
    else:
        cfg.source = 'Moon.mp4'  # 동영상 파일 경로로 변경
        
    cfg.imgsz = 640
    cfg.show    = True    
    cfg.iou     = 0.45
    cfg.conf    = 0.15
    cfg.data    = "coco128.yaml"
    cfg.model   = 'yolov8n.pt'

    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=cfg.model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(cfg.model)(**args)(cfg)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict(use_python = True)
lo_jetson

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
```
 pip install -U pip wheel gdown
 gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
 gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV 입력

```
```

sudo apt-get install libopenblas-base libopenmpi-dev
sudo apt-get install libomp-dev
pip install torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl
pip install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
python -c "import torch; print(torch.__version__)"

```
```
 conda install numpy                                       # 또는 >>> 다음에 설치를 해도 된다.

```
```



(yolo) dli@dli:~$ python

>>> import torch
>>> import torchvision
>>> print(torch.__version__)
>>> print(torchvision.__version__)
>>> print("cuda used", torch.cuda.is_available())
cuda used True
>>>

```
```
git clone https://github.com/Tory-Hwang/Jetson-Nano2

```
```

(yolo) dli@dli:~$ cd Jetson-Nano2/
(yolo) dli@dli:~/Jetson-Nano2$ cd V8
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install ultralytics
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install -r requirements.txt 
(yolo) dli@jdli:~/Jetson-Nano2/V8$ pip install ffmpeg-python
(yolo) dli@dli:~/Jetson-Nano2$ sudo apt install tree
(yolo) dli@jdli:~/Jetson-Nano2$ tree -L 2


```
```


gedit  detectY8.py

```
```

(yolo) dli@jetson:~/Jetson-Nano2/V8$ python detectY8.py

```
```

def predict(cfg=DEFAULT_CFG, use_python=False): brtsp = True -> brtsp = False detectY8.py 스크립트에서 RTSP(Remote Desktop Protocol)를 사용하지 않도록 변경하려면, brtsp = True 라인을 찾아 False로 변경해야 함
이 변경은 스크립트가 RTSP 스트리밍 대신 다른 비디오 입력 소스(예: USB 카메라)를 사용하도록 지시함
코드에서 brtsp 변수가 RTSP 스트리밍을 활성화하거나 비활성화하는 데 사용됨

brtsp(Better RTSP) 설정의 목적:더 효율적인 RTSP 스트림 처리 버퍼링 최적화 프레임 스킵 방지 메모리 사용량 최적화

True로 설정할 때의 장점:더 안정적인 스트림 처리 지연시간 감소 메모리 효율성 향상

False로 변경하는 주요 이유들:하드웨어 리소스가 제한적일 때 실시간성보다 처리의 정확성이 더 중요할 때 메모리 사용량을 줄여야 할 때 레거시 시스템과의 호환성이 필요할 때

```
```


(yolo) yolo@yolo-desktop:~/Jetson-Nano2/V8$ python detectY8.py

```

```
# Ultralytics YOLO , GPL-3.0 license

import argparse
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
   
    brtsp = False
    if brtsp: 
        cfg.source = 'rtsp://admin:satech1234@192.168.0.151:554/udp/av0_0'
    else:
        cfg.source = 'Moon.mp4'  # 동영상 파일 경로로 변경
        
    cfg.imgsz = 640
    cfg.show    = True    
    cfg.iou     = 0.45
    cfg.conf    = 0.15
    cfg.data    = "coco128.yaml"
    cfg.model   = 'yolov8n.pt'

    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=cfg.model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(cfg.model)(**args)(cfg)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict(use_python = True)
 yo# yolo_jetson

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
```
 pip install -U pip wheel gdown
 gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
 gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV 입력

```
```

sudo apt-get install libopenblas-base libopenmpi-dev
sudo apt-get install libomp-dev
pip install torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl
pip install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
python -c "import torch; print(torch.__version__)"

```
```
 conda install numpy                                       # 또는 >>> 다음에 설치를 해도 된다.

```
```



(yolo) dli@dli:~$ python

>>> import torch
>>> import torchvision
>>> print(torch.__version__)
>>> print(torchvision.__version__)
>>> print("cuda used", torch.cuda.is_available())
cuda used True
>>>

```
```
git clone https://github.com/Tory-Hwang/Jetson-Nano2

```
```

(yolo) dli@dli:~$ cd Jetson-Nano2/
(yolo) dli@dli:~/Jetson-Nano2$ cd V8
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install ultralytics
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install -r requirements.txt 
(yolo) dli@jdli:~/Jetson-Nano2/V8$ pip install ffmpeg-python
(yolo) dli@dli:~/Jetson-Nano2$ sudo apt install tree
(yolo) dli@jdli:~/Jetson-Nano2$ tree -L 2


```
```


gedit  detectY8.py

```
```

(yolo) dli@jetson:~/Jetson-Nano2/V8$ python detectY8.py

```
```

def predict(cfg=DEFAULT_CFG, use_python=False): brtsp = True -> brtsp = False detectY8.py 스크립트에서 RTSP(Remote Desktop Protocol)를 사용하지 않도록 변경하려면, brtsp = True 라인을 찾아 False로 변경해야 함
이 변경은 스크립트가 RTSP 스트리밍 대신 다른 비디오 입력 소스(예: USB 카메라)를 사용하도록 지시함
코드에서 brtsp 변수가 RTSP 스트리밍을 활성화하거나 비활성화하는 데 사용됨

brtsp(Better RTSP) 설정의 목적:더 효율적인 RTSP 스트림 처리 버퍼링 최적화 프레임 스킵 방지 메모리 사용량 최적화

True로 설정할 때의 장점:더 안정적인 스트림 처리 지연시간 감소 메모리 효율성 향상

False로 변경하는 주요 이유들:하드웨어 리소스가 제한적일 때 실시간성보다 처리의 정확성이 더 중요할 때 메모리 사용량을 줄여야 할 때 레거시 시스템과의 호환성이 필요할 때

```
```


(yolo) yolo@yolo-desktop:~/Jetson-Nano2/V8$ python detectY8.py

```

```
# Ultralytics YOLO , GPL-3.0 license

import argparse
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
   
    brtsp = False
    if brtsp: 
        cfg.source = 'rtsp://admin:satech1234@192.168.0.151:554/udp/av0_0'
    else:
        cfg.source = 'Moon.mp4'  # 동영상 파일 경로로 변경
        
    cfg.imgsz = 640
    cfg.show    = True    
    cfg.iou     = 0.45
    cfg.conf    = 0.15
    cfg.data    = "coco128.yaml"
    cfg.model   = 'yolov8n.pt'

    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=cfg.model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(cfg.model)(**args)(cfg)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict(use_python = True)
lo_jetson

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
conda create -n yolo python=3.8 -y 입 력
결과
새로운 패키지 다운로드
```
```
 conda activate yolo 입력
결과 
(yolo) dli@dli-desktop:~$
```
```
 pip install -U pip wheel gdown
 gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
 gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV 입력

```
```

sudo apt-get install libopenblas-base libopenmpi-dev
sudo apt-get install libomp-dev
pip install torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl
pip install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
python -c "import torch; print(torch.__version__)"

```
```
 conda install numpy                                      

```
```



(yolo) dli@dli:~$ python

>>> import torch
>>> import torchvision
>>> print(torch.__version__)
>>> print(torchvision.__version__)
>>> print("cuda used", torch.cuda.is_available())
cuda used True
>>>

```
```
git clone https://github.com/Tory-Hwang/Jetson-Nano2

```
```

(yolo) dli@dli:~$ cd Jetson-Nano2/
(yolo) dli@dli:~/Jetson-Nano2$ cd V8
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install ultralytics
(yolo) dli@dli:~/Jetson-Nano2/V8$ pip install -r requirements.txt 
(yolo) dli@jdli:~/Jetson-Nano2/V8$ pip install ffmpeg-python
(yolo) dli@dli:~/Jetson-Nano2$ sudo apt install tree
(yolo) dli@jdli:~/Jetson-Nano2$ tree -L 2


```
```


gedit  detectY8.py

```
```

(yolo) dli@jetson:~/Jetson-Nano2/V8$ python detectY8.py

```
```

def predict(cfg=DEFAULT_CFG, use_python=False): brtsp = True -> brtsp = False detectY8.py 스크립트에서 RTSP(Remote Desktop Protocol)를 사용하지 않도록 변경하려면, brtsp = True 라인을 찾아 False로 변경해야 함
이 변경은 스크립트가 RTSP 스트리밍 대신 다른 비디오 입력 소스(예: USB 카메라)를 사용하도록 지시함
코드에서 brtsp 변수가 RTSP 스트리밍을 활성화하거나 비활성화하는 데 사용됨

brtsp(Better RTSP) 설정의 목적:더 효율적인 RTSP 스트림 처리 버퍼링 최적화 프레임 스킵 방지 메모리 사용량 최적화

True로 설정할 때의 장점:더 안정적인 스트림 처리 지연시간 감소 메모리 효율성 향상

False로 변경하는 주요 이유들:하드웨어 리소스가 제한적일 때 실시간성보다 처리의 정확성이 더 중요할 때 메모리 사용량을 줄여야 할 때 레거시 시스템과의 호환성이 필요할 때

```
```


(yolo) yolo@yolo-desktop:~/Jetson-Nano2/V8$ python detectY8.py

```

```
/b Ultralytics YOLO , GPL-3.0 license

import argparse
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
   
    brtsp = False
    if brtsp: 
        cfg.source = 'rtsp://admin:satech1234@192.168.0.151:554/udp/av0_0'
    else:
        cfg.source = 'Moon.mp4'  # 동영상 파일 경로로 변경
        
    cfg.imgsz = 640
    cfg.show    = True    
    cfg.iou     = 0.45
    cfg.conf    = 0.15
    cfg.data    = "coco128.yaml"
    cfg.model   = 'yolov8n.pt'

    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=cfg.model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(cfg.model)(**args)(cfg)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict(use_python = True)
