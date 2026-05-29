import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    print("=" * 50)
    print("Đánh giá Model trên tập Test")
    print("=" * 50)

    model_path = MODEL_DIR / "cats_dogs_resnet50.h5"
    if not model_path.exists():
        print(f"Không tìm thấy model tại {model_path}")
        print("Chạy 'python train.py' trước.")
        return

    print("\n[1/3] Load model...")
    model = load_model(model_path)

    print("\n[2/3] Chuẩn bị test data...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    print(f"  Test: {test_generator.samples} ảnh")
    print(f"  Classes: {test_generator.class_indices}")

    print("\n[3/3] Đánh giá...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"\n  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Report đã lưu tại: {RESULTS_DIR / 'classification_report.txt'}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix - Cats vs Dogs")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix đã lưu tại: {RESULTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
