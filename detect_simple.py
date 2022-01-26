import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
import sys

MODEL_PATH = "./checkpoints/yolov4-416"
# 예측한 객체에 대한 정보를 가지는 Bounding Box와 정담 Box의 임계값이 0.45 이상일 경우 일치하는 객체로 판단
IOU_THRESHOLD = 0.45
# Threshold for minimum confidence the model has on a detected object
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures["serving_default"]


def main(img_path, output_dir):

    print("start!!", img_path, output_dir)
    # 이미지 로드
    img = cv2.imread(img_path)
    image_show = Image.fromarray(img.astype(np.uint8))
    # image_show.show()
    # img.imshow()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 이미지 크기 변경 416 * 416 으로
    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_input = img_input / 255

    # 차원 하나 추가
    img_input = img_input[np.newaxis, ...].astype(np.float32)

    # numpy array 를 tensor로
    img_input = tf.constant(img_input)

    # prediction한 바운딩 박스
    pred_bbox = infer(img_input)

    # 바운딩 박스를 후처리해서
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # 후처리할때 pressing하는데 NMX를 써서 저 threshold를 넘는애들만 나오게 하고

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD,
    )

    pred_bbox = [
        boxes.numpy(),
        scores.numpy(),
        classes.numpy(),
        valid_detections.numpy(),
    ]
    print(pred_bbox)
    result = utils.draw_bbox(img, pred_bbox)

    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_dir, result)


if __name__ == "__main__":
    print("start detect object")
    img_path = sys.argv[1]
    output_dir = sys.argv[2]
    main(img_path, output_dir)
