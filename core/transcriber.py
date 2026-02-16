"""
Модуль для распознавания речи из аудио с помощью Whisper
"""
import logging
import torch

# Корректный импорт для OpenAI Whisper
try:
    import whisper
except ImportError:
    logging.error("Не удалось импортировать whisper. Установите пакет openai-whisper.")
    whisper = None
    raise RuntimeError("Whisper не установлен")

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    Класс для распознавания речи с помощью модели Whisper
    """
    def __init__(self, whisper_model_size='medium', language='en-US'):
        """
        Инициализация модели Whisper
        
        Args:
            whisper_model_size (str): Размер модели Whisper (medium, base, small, medium, large)
            language (str): Код языка аудио
        """
        self.model_size = whisper_model_size
        self.language = language
        self.model = self._init_model()
        
    def _init_model(self):
        """Инициализация модели Whisper для распознавания речи"""
        if whisper is None:
            logger.error("Whisper не доступен. Убедитесь, что пакет openai-whisper установлен.")
            raise RuntimeError("Whisper не установлен")
            
        logger.info(f"Загрузка модели Whisper {self.model_size}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(self.model_size, device=device)
        
        logger.info(f"Модель Whisper загружена, используется {device}")
        return model
    
    def transcribe(self, audio_path):
        """
        Транскрипция аудио в текст
        
        Args:
            audio_path (str): Путь к аудиофайлу
            
        Returns:
            str: Распознанный текст
        """
        logger.info("Начинаем распознавание речи с помощью Whisper...")
        
        # Используем только Whisper для распознавания
        result = self.model.transcribe(
            audio_path, 
            language=self.language.split('-')[0]
        )
        transcript = result["text"]
        
        logger.info(f"Распознавание речи завершено. Длина текста: {len(transcript)}")
        return transcript 