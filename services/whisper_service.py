"""
Сервис для распознавания речи с использованием OpenAI Whisper
"""

import os
import logging
import whisper
import torch
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class WhisperService:
    """
    Сервис для распознавания речи с использованием OpenAI Whisper
    Предоставляет высококачественное распознавание речи с поддержкой множества языков
    """
    
    def __init__(self, model_name: str = "medium", device: str = None):
        """
        Инициализация сервиса Whisper
        
        Args:
            model_name (str): Название модели Whisper ('tiny', 'base', 'small', 'medium', 'large')
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.model_name = model_name
        
        # Определяем устройство для вычислений
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"Инициализация Whisper модели '{model_name}' на устройстве {device}")
        try:
            self.model = whisper.load_model(model_name).to(device)
            logger.info(f"Модель Whisper '{model_name}' успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели Whisper: {str(e)}")
            raise
    
    def transcribe(self, audio_path: str, language: str = None, task: str = "transcribe") -> Dict[str, Any]:
        """
        Распознавание речи из аудио файла
        
        Args:
            audio_path (str): Путь к аудио файлу
            language (str): Код языка (например, 'en' для английского)
            task (str): Задача ('transcribe' или 'translate')
            
        Returns:
            Dict[str, Any]: Результат распознавания, включая текст и сегменты
        """
        try:
            logger.info(f"Начало распознавания аудио файла: {audio_path}")
            
            # Проверяем существование файла
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
            
            # Настраиваем параметры распознавания
            options = {
                "task": task,
                "verbose": True
            }
            if language:
                options["language"] = language
            
            # Выполняем распознавание
            result = self.model.transcribe(audio_path, **options)
            
            logger.info(f"Распознавание завершено успешно: {len(result['text'])} символов")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при распознавании аудио: {str(e)}")
            raise
    
    def transcribe_segments(self, audio_path: str, language: str = None) -> List[Dict[str, Any]]:
        """
        Распознавание речи с разбивкой на сегменты
        
        Args:
            audio_path (str): Путь к аудио файлу
            language (str): Код языка
            
        Returns:
            List[Dict[str, Any]]: Список сегментов с распознанным текстом
        """
        result = self.transcribe(audio_path, language)
        return result.get("segments", [])
    
    def detect_language(self, audio_path: str) -> str:
        """
        Определение языка речи в аудио файле
        
        Args:
            audio_path (str): Путь к аудио файлу
            
        Returns:
            str: Код определенного языка
        """
        try:
            logger.info(f"Определение языка для аудио: {audio_path}")
            
            # Загружаем аудио и определяем язык
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Получаем mel-спектрограмму
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Определяем язык
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            logger.info(f"Определен язык: {detected_lang} (вероятность: {probs[detected_lang]:.2%})")
            return detected_lang
            
        except Exception as e:
            logger.error(f"Ошибка при определении языка: {str(e)}")
            raise
    
    @property
    def available_languages(self) -> List[str]:
        """
        Получение списка поддерживаемых языков
        
        Returns:
            List[str]: Список кодов поддерживаемых языков
        """
        return list(whisper.tokenizer.LANGUAGES.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о текущей модели
        
        Returns:
            Dict[str, Any]: Информация о модели
        """
        return {
            "name": self.model_name,
            "device": self.device,
            "is_multilingual": True,
            "parameters": self.model.dims.__dict__
        } 