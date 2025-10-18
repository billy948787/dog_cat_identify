import tensorflow as tf
import numpy as np
import albumentations as A
from PIL import Image
from tensorflow import keras
from ultralytics import YOLO
import os

# 硬編碼配置
MODEL_PATH = "model_fold_10.keras"
TEST_IMAGE_PATH = "own_test_image/test1.jpg"  # 單張圖片路徑
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['Cat', 'Dog']

# 載入模型
yolo_model = YOLO('yolo11n.pt')
model = keras.models.load_model(MODEL_PATH)

# 基礎轉換
base_transform = A.Compose([
    A.Resize(width=IMG_WIDTH, height=IMG_HEIGHT),
])

def predict_image(image_path):
    """預測單張圖片"""
    # 讀取並預處理
    img = Image.open(image_path).convert("RGB")
    result = yolo_model.predict(img, conf=0.3, show_labels=False, 
                               visualize=False, verbose=False)[0]
    img_array = result.plot(labels=False)
    img_array = np.array(img_array, dtype=np.float32) / 255.0
    img_array = base_transform(image=img_array)['image']
    
    # 預測
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    
    return predicted_class

# 執行預測
if __name__ == "__main__":
    result = predict_image(TEST_IMAGE_PATH)
    print(result)