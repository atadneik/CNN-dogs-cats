import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Cat", "Dog"]


def predict_image(image_path, model):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx] * 100

    return CLASS_NAMES[class_idx], confidence, img


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python predict.py <đường_dẫn_ảnh>")
        print("Ví dụ:     python predict.py data/test/cats/cat.100.jpg")
        return

    model_path = MODEL_DIR / "cats_dogs_resnet50.h5"
    if not model_path.exists():
        print(f"Không tìm thấy model tại {model_path}")
        print("Chạy 'python train.py' trước.")
        return

    model = load_model(model_path)

    for image_path in sys.argv[1:]:
        if not Path(image_path).exists():
            print(f"Không tìm thấy ảnh: {image_path}")
            continue

        label, confidence, img = predict_image(image_path, model)
        print(f"\n{image_path}")
        print(f"  Kết quả: {label} ({confidence:.1f}%)")

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"{label} - {confidence:.1f}%", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
