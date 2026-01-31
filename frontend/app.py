from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
from datetime import datetime
import json
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import atexit
from fpdf import FPDF

from database import db, DetectionResult
from ai_processor import AIProcessor

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Создаем директории
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Инициализация
db.init_app(app)
ai_processor = AIProcessor()

# Создаем таблицы при запуске приложения
with app.app_context():
    db.create_all()


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Загрузка и обработка изображения"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Проверяем расширение файла
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400

    # Сохраняем файл
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = file.filename.rsplit('.', 1)[0]
    extension = file.filename.rsplit('.', 1)[1].lower()
    filename = f'{timestamp}_{original_filename}.{extension}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Обрабатываем изображение
        result = ai_processor.process_image(filepath)

        # Сохраняем в БД
        detection = DetectionResult(
            filename=filename,
            violation_detected=result['violation'],
            violation_ratio=result.get('violation_ratio', 0),
            total_motorcycles=result['total_motorcycles'],
            processing_time=result['processing_time'],
            bounding_boxes=json.dumps(result.get('motorcycles', [])),
            image_path=result.get('annotated_image', '')
        )
        db.session.add(detection)
        db.session.commit()

        # Добавляем ID к результату
        result['detection_id'] = detection.id
        if result.get('annotated_image'):
            result['annotated_image_url'] = f"/{result['annotated_image']}"
        else:
            result['annotated_image_url'] = ''

        return jsonify(result)

    except Exception as e:
        # Удаляем временный файл в случае ошибки
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Страница истории запросов"""
    page = request.args.get('page', 1, type=int)
    per_page = 10

    detections = DetectionResult.query.order_by(DetectionResult.upload_time.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    return render_template('history.html', detections=detections)


@app.route('/api/history')
def api_history():
    """API для получения истории"""
    detections = DetectionResult.query.order_by(DetectionResult.upload_time.desc()).all()
    return jsonify([d.to_dict() for d in detections])


@app.route('/report/pdf')
def generate_pdf():
    """Генерация PDF отчета с использованием FPDF"""
    try:
        detections = DetectionResult.query.order_by(DetectionResult.upload_time.desc()).all()

        pdf = FPDF()
        pdf.add_page()

        # Заголовок
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "AI Detection System Report", ln=True, align='C')
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)

        if not detections:
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, "No detection data available.", ln=True, align='C')
        else:
            # Сводная статистика
            total = len(detections)
            violations = len([d for d in detections if d.violation_detected])
            total_motorcycles = sum([d.total_motorcycles for d in detections])
            avg_time = sum([d.processing_time for d in detections]) / total if total > 0 else 0

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Summary Statistics", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Total Detections: {total}", ln=True)
            pdf.cell(0, 8, f"Violations Found: {violations}", ln=True)
            pdf.cell(0, 8, f"Violation Rate: {violations / total * 100:.1f}%" if total > 0 else "Violation Rate: 0%",
                     ln=True)
            pdf.cell(0, 8, f"Total Motorcycles Detected: {total_motorcycles}", ln=True)
            pdf.cell(0, 8, f"Average Processing Time: {avg_time:.2f} seconds", ln=True)

            pdf.ln(10)

            # Таблица деталей
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Detection Details", ln=True)
            pdf.ln(5)

            # Заголовки таблицы
            pdf.set_font("Arial", 'B', 10)
            col_widths = [12, 40, 25, 20, 25, 25, 30]
            headers = ['ID', 'Filename', 'Time', 'Violation', 'Ratio', 'Motos', 'Time(s)']

            # Рисуем заголовки
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1, align='C')
            pdf.ln()

            # Данные таблицы
            pdf.set_font("Arial", '', 9)
            for detection in detections[:100]:  # Ограничиваем 100 записями
                # Подготовка данных для строки
                filename = detection.filename
                if len(filename) > 15:
                    filename = filename[:12] + "..."

                time_str = detection.upload_time.strftime('%H:%M') if detection.upload_time else "N/A"
                violation = "YES" if detection.violation_detected else "NO"
                ratio = f"{detection.violation_ratio:.2f}"
                motos = str(detection.total_motorcycles)
                proc_time = f"{detection.processing_time:.1f}"

                # Рисуем строку
                pdf.cell(col_widths[0], 8, str(detection.id), border=1, align='C')
                pdf.cell(col_widths[1], 8, filename, border=1)
                pdf.cell(col_widths[2], 8, time_str, border=1, align='C')
                pdf.cell(col_widths[3], 8, violation, border=1, align='C')
                pdf.cell(col_widths[4], 8, ratio, border=1, align='C')
                pdf.cell(col_widths[5], 8, motos, border=1, align='C')
                pdf.cell(col_widths[6], 8, proc_time, border=1, align='C')
                pdf.ln()

            # Если записей больше 100, показываем сообщение
            if len(detections) > 100:
                pdf.ln(5)
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 8, f"Showing 100 out of {len(detections)} records", ln=True)

            # График статистики по дням (если есть данные)
            if len(detections) > 1:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Daily Statistics", ln=True)
                pdf.ln(5)

                # Группируем по дням
                from collections import defaultdict
                daily_stats = defaultdict(lambda: {'count': 0, 'violations': 0})

                for detection in detections:
                    if detection.upload_time:
                        day = detection.upload_time.strftime('%Y-%m-%d')
                        daily_stats[day]['count'] += 1
                        if detection.violation_detected:
                            daily_stats[day]['violations'] += 1

                # Таблица по дням
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(40, 8, "Date", border=1)
                pdf.cell(30, 8, "Total", border=1, align='C')
                pdf.cell(30, 8, "Violations", border=1, align='C')
                pdf.cell(40, 8, "Violation %", border=1, align='C')
                pdf.ln()

                pdf.set_font("Arial", '', 9)
                for day in sorted(daily_stats.keys(), reverse=True)[:30]:  # Последние 30 дней
                    stats = daily_stats[day]
                    violation_pct = (stats['violations'] / stats['count'] * 100) if stats['count'] > 0 else 0

                    pdf.cell(40, 8, day, border=1)
                    pdf.cell(30, 8, str(stats['count']), border=1, align='C')
                    pdf.cell(30, 8, str(stats['violations']), border=1, align='C')
                    pdf.cell(40, 8, f"{violation_pct:.1f}%", border=1, align='C')
                    pdf.ln()

        # Сохраняем в буфер
        buffer = BytesIO()
        pdf_output = pdf.output()
        buffer.write(pdf_output)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"PDF generation error: {e}")
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500


@app.route('/report/excel')
def generate_excel():
    """Генерация Excel отчета"""
    try:
        detections = DetectionResult.query.order_by(DetectionResult.upload_time.desc()).all()

        if not detections:
            # Создаем пустой отчет
            buffer = BytesIO()
            buffer.write(b"No data available")
            buffer.seek(0)

            return send_file(
                buffer,
                as_attachment=True,
                download_name=f"detection_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mimetype='text/plain'
            )

        # Создаем DataFrame
        data = []
        for d in detections:
            data.append(d.to_dict())

        df = pd.DataFrame(data)

        # Сохраняем в BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detections', index=False)

            # Добавляем лист со статистикой
            summary_data = {
                'Metric': ['Total Detections', 'Violations', 'No Violations', 'Total Motorcycles',
                           'Avg Processing Time'],
                'Value': [
                    len(detections),
                    len([d for d in detections if d.violation_detected]),
                    len([d for d in detections if not d.violation_detected]),
                    sum([d.total_motorcycles for d in detections]),
                    sum([d.processing_time for d in detections]) / len(detections) if detections else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"detection_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        app.logger.error(f"Excel generation error: {e}")
        return jsonify({'error': f'Failed to generate Excel: {str(e)}'}), 500


@app.route('/delete/<int:id>', methods=['DELETE'])
def delete_record(id):
    """Удаление записи"""
    try:
        detection = DetectionResult.query.get_or_404(id)

        # Удаляем связанные файлы изображений
        if detection.image_path and os.path.exists(detection.image_path):
            os.remove(detection.image_path)

        db.session.delete(detection)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def send_static(path):
    """Отдача статических файлов"""
    return send_from_directory('static', path)


# Функция для очистки временных файлов при выходе
def cleanup_temp_files():
    import glob
    import time

    # Удаляем файлы старше 24 часов
    current_time = time.time()
    for folder in ['static/uploads', 'static/results']:
        for file in glob.glob(os.path.join(folder, '*')):
            if os.path.getmtime(file) < current_time - 24 * 3600:
                try:
                    os.remove(file)
                except:
                    pass


# Регистрируем функцию очистки
atexit.register(cleanup_temp_files)


# Простая страница с ошибкой
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Страница не найдена'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Внутренняя ошибка сервера'), 500


if __name__ == '__main__':
    print("Starting AI Detection System...")
    print("Open http://localhost:5000 in your browser")

    # Проверяем существование моделей
    model_files = ['models/main_model.pt', 'models/yolo11n-seg.pt']
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"WARNING: Model file not found: {model_file}")

    app.run(debug=True, port=5000)