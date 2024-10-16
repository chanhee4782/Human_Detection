import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from PIL import Image


# 이미지 디렉토리와 레이블 디렉토리 경로 설정
image_dir = 'Test_img/img'
label_dir = 'Test_img/img_label'

# YOLO 레이블 파일에서 정규화된 바운딩 박스 좌표를 읽는 함수
def load_labels(label_file):
    boxes = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            # YOLO 형식: class_id, x_center, y_center, width, height
            class_id, x_center, y_center, width, height = map(float, data)
            boxes.append([class_id, x_center, y_center, width, height])  # 정규화된 좌표를 그대로 사용
    return boxes

# 이미지 및 레이블 데이터를 로드하는 함수
def load_data(image_dir, label_dir, img_width, img_height, max_boxes=10):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    label_files = sorted([os.path.join(label_dir, f.replace('.jpg', '.txt')) for f in os.listdir(image_dir) if f.endswith('.jpg')])

    images = []
    labels = []
    missing_labels = 0

    for img_file, lbl_file in zip(image_files, label_files):
        # 레이블 파일 존재 여부 확인
        if not os.path.exists(lbl_file):
            print(f"Label file not found: {lbl_file}")
            missing_labels += 1
            continue
        
        # 이미지 읽기
        img = tf.keras.preprocessing.image.load_img(img_file, target_size=(img_height, img_width))
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img)

        # 레이블 읽기
        boxes = load_labels(lbl_file)
        
        # 바운딩 박스 수가 max_boxes보다 적은 경우, None으로 패딩
        while len(boxes) < max_boxes:
            boxes.append([0, 0, 0, 0, 0])  # 클래스 ID와 좌표를 0으로 채움

        labels.append(boxes[:max_boxes])  # max_boxes로 잘라냄

    print(f"Missing label files: {missing_labels}")
    return np.array(images), np.array(labels)

# 이미지 및 레이블 로드
img_width = 32
img_height = 32

train_images, train_labels = load_data(image_dir, label_dir, img_width, img_height)

# 텐서플로 데이터셋으로 변환
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)

# 0-------------------
def create_cnn_model(input_shape, max_boxes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(max_boxes * 5)  # max_boxes 수에 맞게 출력 차원 설정
    ])
    return model

# 커스텀 레이어로 출력을 reshape
class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, target_shape):
        super(ReshapeLayer, self).__init__()
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, (-1, *self.target_shape))

# 모델 생성
img_height, img_width = 32, 32
max_boxes = 10  # 최대 바운딩 박스 수
input_shape = (img_height, img_width, 3)  # RGB 이미지

model = create_cnn_model(input_shape, max_boxes)

# reshape 레이어 추가
model.add(ReshapeLayer((max_boxes, 5)))  # 출력 형태를 (batch_size, max_boxes, 5)로 변경

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# 모델 학습
model.fit(train_ds, epochs=500)  # 에포크 수는 조정 가능

# 학습 완료 메시지 출력
print("훈련 완료!")