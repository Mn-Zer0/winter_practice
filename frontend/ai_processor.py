import cv2
import numpy as np
from ultralytics import YOLO
import time
import json


class AIProcessor:
    def __init__(self):
        self.model_A = YOLO("models/main_model.pt")
        self.model_seg = YOLO("models/yolo11n-seg.pt")
        self.SEG_THRESHOLD = 0.1

    def process_image(self, image_path):
        """Основная логика обработки изображения"""
        start_time = time.time()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        h, w = img.shape[:2]
        overlay = img.copy()

        # Результаты для сохранения
        result = {
            'violation': False,
            'violation_ratio': 0,
            'total_motorcycles': 0,
            'motorcycles': [],
            'letter_a_box': None,
            'line_points': None
        }

        # 1. DETECT LETTER A
        res_A = self.model_A(img, conf=0.3)[0]

        if res_A.boxes is None or len(res_A.boxes) == 0:
            return self._create_result(result, start_time, "Letter A not found")

        ax1, ay1, ax2, ay2 = res_A.boxes.xyxy[0].cpu().numpy().astype(int)
        a_cx = (ax1 + ax2) // 2
        a_cy = (ay1 + ay2) // 2

        result['letter_a_box'] = [int(ax1), int(ay1), int(ax2), int(ay2)]

        # 2. ROI AROUND LETTER A
        rx1 = max(a_cx - 320, 0)
        rx2 = min(a_cx + 320, w)
        ry1 = max(ay1 - 40, 0)
        ry2 = min(ay2 + 200, h)

        roi = img[ry1:ry2, rx1:rx2]

        # 3. FIND WHITE LINE
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(white, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=20
        )

        if lines is None:
            return self._create_result(result, start_time, "White line not found")

        # 4. FIND LINE NEAR A
        best_line = None
        best_dist = 1e9

        for l in lines:
            x1, y1, x2, y2 = l[0]
            gx1, gy1 = x1 + rx1, y1 + ry1
            gx2, gy2 = x2 + rx1, y2 + ry1

            if max(gy1, gy2) < ay1 or min(gy1, gy2) > ay2:
                continue

            if gx2 < ax1:
                dist = ax1 - max(gx1, gx2)
            elif gx1 > ax2:
                dist = min(gx1, gx2) - ax2
            else:
                continue

            if dist < best_dist:
                best_dist = dist
                best_line = (gx1, gy1, gx2, gy2)

        if best_line is None:
            return self._create_result(result, start_time, "Line near A not found")

        # 5. EXTEND LINE
        lx1, ly1, lx2, ly2 = best_line

        if lx1 == lx2:
            pt1 = (lx1, 0)
            pt2 = (lx1, h)
        else:
            k = (ly2 - ly1) / (lx2 - lx1)
            b = ly1 - k * lx1
            pt1 = (0, int(b))
            pt2 = (w, int(k * w + b))

        result['line_points'] = [int(lx1), int(ly1), int(lx2), int(ly2)]

        # 6. SIDE FUNCTION
        def side_of_line(px, py, x1, y1, x2, y2):
            return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

        a_side = side_of_line(a_cx, a_cy, lx1, ly1, lx2, ly2)

        # 7. DETECT MOTORCYCLES
        res = self.model_seg(img, conf=0.3)[0]
        motorcycle_count = 0

        if res.masks is not None:
            for i, (box, cls) in enumerate(zip(res.boxes.xyxy, res.boxes.cls)):
                if int(cls) != 3:  # 3 - класс motorcycle в COCO
                    continue

                motorcycle_count += 1
                mx1, my1, mx2, my2 = box.cpu().numpy().astype(int)
                poly = res.masks.xy[i]

                if poly.shape[0] < 3:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                poly_int = poly.astype(np.int32)
                cv2.fillPoly(mask, [poly_int], 1)

                ys, xs = np.where(mask == 1)
                total = len(xs)

                if total == 0:
                    continue

                same_side = 0
                for x, y in zip(xs, ys):
                    m_side = side_of_line(x, y, lx1, ly1, lx2, ly2)
                    if m_side == 0 or (m_side > 0 and a_side > 0) or (m_side < 0 and a_side < 0):
                        same_side += 1

                ratio = same_side / total
                is_violation = ratio >= self.SEG_THRESHOLD

                if is_violation:
                    result['violation'] = True
                    result['violation_ratio'] = max(result['violation_ratio'], ratio)

                motorcycle_data = {
                    'bbox': [int(mx1), int(my1), int(mx2), int(my2)],
                    'violation': bool(is_violation),
                    'violation_ratio': float(ratio)
                }
                result['motorcycles'].append(motorcycle_data)

        result['total_motorcycles'] = motorcycle_count

        # Сохраняем обработанное изображение с аннотациями
        output_path = self._save_annotated_image(img, overlay, result, image_path)
        result['annotated_image'] = output_path

        return self._create_result(result, start_time)

    def _create_result(self, result, start_time, error=None):
        """Создание финального результата"""
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        result['success'] = error is None
        result['error'] = error
        return result

    def _save_annotated_image(self, original, overlay, result, original_path):
        """Сохранение изображения с аннотациями"""
        import os
        from datetime import datetime

        # Создаем директорию для результатов
        results_dir = 'static/results'
        os.makedirs(results_dir, exist_ok=True)

        # Генерируем имя файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'result_{timestamp}.jpg'
        output_path = os.path.join(results_dir, filename)

        # Рисуем аннотации
        img_copy = overlay.copy()

        # Рисуем букву A
        if result['letter_a_box']:
            ax1, ay1, ax2, ay2 = result['letter_a_box']
            cv2.rectangle(img_copy, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
            cv2.putText(img_copy, 'Letter A', (ax1, ay1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Рисуем линию
        if result['line_points']:
            lx1, ly1, lx2, ly2 = result['line_points']
            cv2.line(img_copy, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)
            cv2.putText(img_copy, 'Boundary Line', (lx1, ly1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Рисуем мотоциклы
        for moto in result['motorcycles']:
            mx1, my1, mx2, my2 = moto['bbox']
            color = (0, 0, 255) if moto['violation'] else (0, 255, 0)
            label = f"Violation {moto['violation_ratio']:.2f}" if moto[
                'violation'] else f"OK {moto['violation_ratio']:.2f}"

            cv2.rectangle(img_copy, (mx1, my1), (mx2, my2), color, 2)
            cv2.putText(img_copy, label, (mx1, my1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Добавляем статистику
        stats = f"Violation: {result['violation']} | Motorcycles: {result['total_motorcycles']}"
        cv2.putText(img_copy, stats, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Сохраняем
        cv2.imwrite(output_path, img_copy)
        return output_path