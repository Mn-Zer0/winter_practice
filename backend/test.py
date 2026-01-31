from ultralytics import YOLO
import cv2
import os


MODEL_PATH = r"runs/detect/train_for_markup/letter_A_train/weights/best.pt"
IMAGE_PATH = r"image_for_test1.jpg"
CONF = 0.3


model = YOLO(MODEL_PATH)


results = model.predict(
    source=IMAGE_PATH,
    conf=CONF,
    save=True,
    show=False
)


img = cv2.imread(IMAGE_PATH)

for r in results:
    for box, score in zip(r.boxes.xyxy, r.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"A {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

cv2.imshow("Letter A detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Готово.")
print("Результат также сохранён в папке runs/detect/predict/")