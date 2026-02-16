"""
Модуль для синтеза речи с использованием Yandex SpeechKit
"""
import os
import logging
from typing import Dict, List
from speechkit import configure_credentials, creds, model_repository

logger = logging.getLogger(__name__)

class YandexTTS:
    """
    Класс для работы с Yandex SpeechKit
    Использует официальную библиотеку
    """
    def __init__(self, api_key: str, voice: str = 'kz_daulet', emotion: str = 'neutral', speed: float = 1.0):
        """
        Инициализация Yandex TTS
        
        Args:
            api_key (str): API ключ от Yandex Cloud
            voice (str): Голос для синтеза (kz_daulet или kz_mayra)
            emotion (str): Эмоция (neutral, good, evil)
            speed (float): Скорость речи (0.1 до 3.0)
        """
        # Настраиваем аутентификацию
        configure_credentials(
            yandex_credentials=creds.YandexCredentials(
                api_key=api_key
            )
        )
        
        # Создаем модель синтеза
        self.model = model_repository.synthesis_model()
        self.model.voice = voice
        self.model.role = emotion  # В SpeechKit эмоция называется role
        self.model.speed = speed
        
        logger.info(f"Инициализирован Yandex TTS (голос: {voice}, эмоция: {emotion}, скорость: {speed})")
    
    def synthesize_to_mp3(self, text: str, output_path: str) -> str:
        """
        Синтез речи и сохранение в MP3
        
        Args:
            text (str): Текст для синтеза
            output_path (str): Путь для сохранения MP3
            
        Returns:
            str: Путь к созданному MP3 файлу
        """
        try:
            logger.debug(f"Отправляем запрос на синтез: {text[:50]}...")
            
            # Создаем директорию если нужно
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Синтезируем речь
            result = self.model.synthesize(text, raw_format=False)  # Получаем AudioSegment
            
            # Сохраняем в MP3
            result.export(output_path, format='mp3')
            
            logger.info(f"Аудио успешно сохранено: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при синтезе речи: {str(e)}")
            raise
    
    @staticmethod
    def get_available_voices() -> Dict[str, Dict[str, str]]:
        """
        Получение списка доступных голосов
        
        Returns:
            Dict[str, Dict[str, str]]: Словарь с информацией о доступных голосах
        """
        return {
            "kz_daulet": {
                "name": "Даулет",
                "gender": "male",
                "description": "Мужской голос для казахского языка"
            },
            "kz_mayra": {
                "name": "Майра",
                "gender": "female",
                "description": "Женский голос для казахского языка"
            }
        }
    
    @staticmethod
    def get_available_emotions() -> List[str]:
        """
        Получение списка доступных эмоций
        
        Returns:
            List[str]: Список доступных эмоций
        """
        return ["neutral", "good", "evil"] 