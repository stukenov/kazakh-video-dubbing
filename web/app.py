"""
Веб-приложение для системы дубляжа видео с английского на казахский
"""
import os
import uuid
from pathlib import Path
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Импортируем нашу систему дубляжа
from core.dubbing import VideoDubber
from utils.config import Config

# Инициализация Flask приложения
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = Config.PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Словарь для хранения статуса обработки задач
processing_tasks = {}

def allowed_file(filename):
    """Проверка допустимого расширения файла"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def process_video_task(input_path, output_path, task_id):
    """Задача для обработки видео в отдельном потоке"""
    try:
        # Обновляем статус
        processing_tasks[task_id]['status'] = 'processing'
        
        # Создаем экземпляр дубляжа и обрабатываем видео
        dubber = VideoDubber()
        dubber.process_video(input_path, output_path)
        
        # Обновляем статус по завершении
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['output_path'] = output_path
    except Exception as e:
        # В случае ошибки
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загрузки файла"""
    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        flash('Файл не выбран')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Если пользователь не выбрал файл
    if file.filename == '':
        flash('Файл не выбран')
        return redirect(request.url)
    
    # Если файл имеет правильное расширение
    if file and allowed_file(file.filename):
        # Генерируем безопасное имя файла с уникальным идентификатором
        filename = secure_filename(file.filename)
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        unique_filename = f"{base_name}_{uuid.uuid4().hex[:8]}{extension}"
        
        # Пути для сохранения файлов
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        output_filename = f"{base_name}_dubbed_{uuid.uuid4().hex[:8]}{extension}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Сохраняем загруженный файл
        file.save(input_path)
        
        # Создаем идентификатор задачи
        task_id = str(uuid.uuid4())
        
        # Добавляем информацию о задаче в словарь
        processing_tasks[task_id] = {
            'input_path': input_path,
            'output_path': output_path,
            'status': 'queued',
            'created_at': time.time(),
            'original_filename': filename,
            'output_filename': output_filename
        }
        
        # Запускаем обработку видео в отдельном потоке
        thread = threading.Thread(
            target=process_video_task,
            args=(input_path, output_path, task_id)
        )
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('task_status', task_id=task_id))
    
    flash('Недопустимый формат файла')
    return redirect(request.url)

@app.route('/task/<task_id>')
def task_status(task_id):
    """Страница статуса задачи"""
    if task_id not in processing_tasks:
        flash('Задача не найдена')
        return redirect(url_for('index'))
    
    return render_template('task.html', task_id=task_id)

@app.route('/api/task/<task_id>')
def api_task_status(task_id):
    """API для получения статуса задачи"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Задача не найдена'}), 404
    
    task = processing_tasks[task_id]
    
    response = {
        'status': task['status'],
        'created_at': task['created_at'],
        'original_filename': task['original_filename'],
    }
    
    if task['status'] == 'completed':
        response['download_url'] = url_for('download_file', filename=os.path.basename(task['output_path']))
    elif task['status'] == 'error':
        response['error'] = task.get('error', 'Неизвестная ошибка')
    
    return jsonify(response)

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание обработанного файла"""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/clean-old-files')
def clean_old_files():
    """Очистка старых файлов (админская функция)"""
    # В реальном приложении здесь должна быть проверка прав доступа
    
    current_time = time.time()
    old_time = current_time - (24 * 60 * 60)  # 24 часа
    
    # Очистка старых задач
    old_tasks = [task_id for task_id, task in processing_tasks.items() 
                if task['created_at'] < old_time]
    
    for task_id in old_tasks:
        task = processing_tasks[task_id]
        
        # Удаляем файлы, если они существуют
        for path in [task['input_path'], task['output_path']]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        # Удаляем задачу из словаря
        del processing_tasks[task_id]
    
    return jsonify({'message': f'Удалено {len(old_tasks)} старых задач'})

if __name__ == '__main__':
    # Создаем необходимые директории, если они не существуют
    for folder in [Config.UPLOAD_FOLDER, Config.PROCESSED_FOLDER, Config.TEMP_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    # Запускаем приложение
    app.run(debug=True, host='0.0.0.0', port=5000) 