# How to run

## Docker version

1. install docker
   https://docs.docker.com/desktop/mac/install/

2. build image to run object detection

```
make build_image
```

3. run with docker image

```
# make run.simple {target-file} {output-file-name}
make run.simple FILENAME=data/kite.jpg OUTPUTFILENAME=kite_observed.jpg
```

## conda version

1. install conda

https://ikaros79.tistory.com/entry/Mac%EC%97%90%EC%84%9C-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-%EC%84%A4%EC%B9%98%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95

2. build conda env

make conda.build.env

3. run virtual environment

conda activate yolo_python

4. run following commands

### image

```
#python detect_simple.py {image dir} {output-file-name}
python detect_simple.py ./data/kite.jpg kite_observed.jpg
```

### video

```
#python detect_simple.py {video dir} {output-file-name}
python detect_video_simple.py ./data/road.mp4 road_detected.mp4
```

# Note for me

## How to create conda environment with file

conda env create --name yolo_python --file environment.yaml

## How to build conda environment

conda create --name yolo_python python=3.6
pip install -r requirements.txt
conda env export > environment.yaml

## run with conda environment

### References

https://github.com/kairess/tensorflow-yolov4-tflite -> most of codes are from this repository. I just changed it just for testing codes

- YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
- [darknet](https://github.com/AlexeyAB/darknet)
- [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
- [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)
