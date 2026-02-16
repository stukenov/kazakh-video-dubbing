"""
Модуль конфигурации для системы дубляжа
"""
import os
from typing import Dict, Any
import torch
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

class Config:
    """
    Класс для хранения конфигурации
    """
    # Общие настройки
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed')
    TEMP_FOLDER = os.path.join(os.getcwd(), 'temp')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

    # API ключи (из переменных окружения для безопасности)
    API_KEYS = {
        'google_cloud': os.getenv('GOOGLE_CLOUD_API_KEY'),
        'nemo': os.getenv('NEMO_API_KEY'),
        'custom_service': os.getenv('CUSTOM_SERVICE_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'yandex_speechkit': os.getenv('YANDEX_SPEECHKIT_API_KEY'),
        'yandex_folder_id': os.getenv('YANDEX_FOLDER_ID')
    }

    def __init__(self):
        """
        Инициализация конфигурации
        """
        # OpenAI API key
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Yandex SpeechKit API key
        self.yandex_speechkit_api_key = os.getenv('YANDEX_SPEECHKIT_API_KEY', '')

        # Настройки для распознавания речи
        self.SPEECH_RECOGNITION = {
            'model': 'medium',  # Модель Whisper
            'language': 'en',  # Язык исходного видео
            'min_segment_length': 1.0,  # Минимальная длина сегмента в секундах
            'max_segment_length': 10.0,  # Максимальная длина сегмента в секундах
            'whisper_model_size': 'medium',  # Размер модели Whisper
            'sample_rate': 16000  # Частота дискретизации для распознавания
        }
        
        # Настройки для перевода
        self.TRANSLATION = {
            'source_language': 'en',  # Исходный язык
            'target_language': 'kk',  # Целевой язык
            'use_chatgpt': True,  # Использовать ли ChatGPT для перевода
            'service': 'google'  # 'google' или 'custom'
        }
        
        # Настройки для синтеза речи
        self.TTS = {
            'service': 'edge',  # 'yandex' или 'edge'
            'voices': {
                'kk': {  # Казахский язык
                    'male': 'kk-KZ-DauletNeural',
                    'female': 'kk-KZ-AigulNeural'
                },
                'ru': {  # Русский язык
                    'male': 'ru-RU-DmitryNeural',
                    'female': 'ru-RU-SvetlanaNeural'
                },
                'en': {  # Английский язык
                    'male': 'en-US-GuyNeural',
                    'female': 'en-US-JennyNeural'
                }
            },
            'voice_gender': 'female',  # 'male' или 'female'
            'speed': 1.0,  # Скорость речи (1.0 = нормальная)
            'pitch': 1.0,  # Высота тона (1.0 = нормальная)
            
            # Специфичные настройки для Edge TTS
            'edge': {
                'rate': '+0%',  # Изменение скорости в процентах
                'volume': '+0%',  # Изменение громкости в процентах
                'pitch': '+0Hz'  # Изменение высоты тона в герцах
            }
        }
        
        # Настройки для обработки видео
        self.VIDEO_PROCESSING = {
            'codec': 'libx264',  # Видеокодек
            'bitrate': '8000k',  # Битрейт видео
            'fps': None,  # Сохранить оригинальный FPS
            'background_volume': 0.1,  # Громкость фонового аудио (0.0-1.0)
            'output_format': 'mp4',  # Формат выходного файла
            'resolution': None  # Сохранить оригинальное разрешение
        }
        
        # Настройки для обработки аудио
        self.AUDIO_PROCESSING = {
            'sample_rate': 24000,  # Частота дискретизации
            'channels': 1,  # Количество каналов
            'format': 'wav',  # Формат аудио файлов
            'codec': 'pcm_s16le',  # Аудиокодек
            'normalize_audio': True,  # Нормализация аудио
            'trim_silence': True  # Обрезка тишины
        }
        
        # Настройки для временных файлов
        self.TEMP = {
            'root_dir': 'temp_files',  # Корневая директория для временных файлов
            'keep_files': True,  # Сохранять ли временные файлы после обработки
            'cleanup_on_error': False  # Удалять ли временные файлы при ошибке
        }
        
        # Настройки субтитров
        self.SUBTITLES = {
            'min_duration': 1.0,  # Минимальная длительность субтитра (секунды)
            'max_duration': 7.0,  # Максимальная длительность субтитра (секунды)
            'min_gap': 0.1,  # Минимальный промежуток между субтитрами (секунды)
            'font_size': 24,  # Размер шрифта
            'font_color': 'white',  # Цвет текста
            'background_color': 'black',  # Цвет фона
            'background_alpha': 0.4,  # Прозрачность фона
            'char_rate': {  # Скорость произношения (символов в секунду)
                'en': (10, 15),  # Диапазон для английского
                'kk': (8, 13)  # Диапазон для казахского
            }
        }
        
        # Настройки сегментации
        self.SEGMENTATION = {
            'min_segment_length': 0.5,  # Минимальная длина сегмента (секунды)
            'max_segment_length': 10.0,  # Максимальная длина сегмента (секунды)
            'energy_threshold': 50,  # Порог энергии для определения речи
            'pause_threshold': 0.5  # Порог паузы между сегментами (секунды)
        }
        
        # Настройки синтеза речи
        self.SPEECH_SYNTHESIS = {
            'voice': 'kk-KZ-DauletNeural',  # Голос для казахского языка
            'rate': '+0%',  # Скорость речи по умолчанию
            'volume': '+0%'  # Громкость по умолчанию
        }

        # Создаем необходимые директории при инициализации
        for folder in [self.UPLOAD_FOLDER, self.PROCESSED_FOLDER, self.TEMP_FOLDER]:
            os.makedirs(folder, exist_ok=True) 