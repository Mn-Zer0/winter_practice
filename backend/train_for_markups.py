from ultralytics import YOLO
import torch


def main():
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Видеокарта не обнаружена")

    DATA_YAML = "data.yaml"
    MODEL_NAME = "yolo11n.pt"

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project="train_for_markup",
        name="letter_A_train",
        pretrained=True
    )

    print("\nОбучение завершено.")


if __name__ == "__main__":
    main()