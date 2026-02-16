"""
Модуль для обработки аудио файлов в системе дубляжа
Включает функционал для работы с аудио файлами, их анализа и обработки
"""

import os
import numpy as np
import soundfile as sf
import librosa
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Класс для обработки аудио файлов
    Предоставляет методы для работы с аудио: загрузка, сохранение,
    анализ характеристик и базовая обработка
    """
    
    def __init__(self, sample_rate=16000):
        """
        Инициализация процессора аудио
        
        Args:
            sample_rate (int): Частота дискретизации для обработки аудио (по умолчанию 16кГц)
        """
        self.sample_rate = sample_rate
        logger.info(f"Инициализирован обработчик аудио с частотой дискретизации {sample_rate}Hz")
    
    def load_audio(self, file_path):
        """
        Загрузка аудио файла
        
        Args:
            file_path (str): Путь к аудио файлу
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.info(f"Загружен аудио файл: {file_path}")
            return audio_data, sr
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио файла {file_path}: {str(e)}")
            raise
    
    def save_audio(self, audio_data, file_path, sample_rate=None):
        """
        Сохранение аудио данных в файл
        
        Args:
            audio_data (numpy.ndarray): Аудио данные
            file_path (str): Путь для сохранения файла
            sample_rate (int, optional): Частота дискретизации
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            sf.write(file_path, audio_data, sample_rate)
            logger.info(f"Сохранен аудио файл: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении аудио файла {file_path}: {str(e)}")
            raise
    
    def get_audio_duration(self, audio_data, sample_rate=None):
        """
        Получение длительности аудио в секундах
        
        Args:
            audio_data (numpy.ndarray): Аудио данные
            sample_rate (int, optional): Частота дискретизации
            
        Returns:
            float: Длительность аудио в секундах
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        return len(audio_data) / sample_rate
    
    def normalize_audio(self, audio_data):
        """
        Нормализация громкости аудио
        
        Args:
            audio_data (numpy.ndarray): Аудио данные
            
        Returns:
            numpy.ndarray: Нормализованные аудио данные
        """
        return librosa.util.normalize(audio_data)
    
    def trim_silence(self, audio_data, threshold_db=-50):
        """
        Обрезка тишины в начале и конце аудио
        
        Args:
            audio_data (numpy.ndarray): Аудио данные
            threshold_db (float): Порог тишины в децибелах
            
        Returns:
            numpy.ndarray: Обрезанные аудио данные
        """
        trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=-threshold_db)
        return trimmed_audio
    
    def adjust_speed(self, audio_data, speed_factor):
        """
        Изменение скорости воспроизведения аудио
        
        Args:
            audio_data (numpy.ndarray): Аудио данные
            speed_factor (float): Коэффициент изменения скорости (>1 для ускорения, <1 для замедления)
            
        Returns:
            numpy.ndarray: Измененные аудио данные
        """
        return librosa.effects.time_stretch(audio_data, rate=speed_factor)
    
    def mix_audio(self, audio1, audio2, weight1=0.5, weight2=0.5):
        """
        Смешивание двух аудио файлов
        
        Args:
            audio1 (numpy.ndarray): Первый аудио файл
            audio2 (numpy.ndarray): Второй аудио файл
            weight1 (float): Вес первого аудио (0-1)
            weight2 (float): Вес второго аудио (0-1)
            
        Returns:
            numpy.ndarray: Смешанное аудио
        """
        # Приводим аудио к одинаковой длине
        if len(audio1) > len(audio2):
            audio1 = audio1[:len(audio2)]
        else:
            audio2 = audio2[:len(audio1)]
            
        return (audio1 * weight1 + audio2 * weight2) 