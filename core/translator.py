"""
Модуль для перевода текста с поддержкой Google Translate и ChatGPT 4
"""
import os
import json
import time
import logging
import certifi
import asyncio
import re
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
from googletrans import Translator as GoogleTranslator
from openai import AsyncOpenAI
from dotenv import load_dotenv
from models.speech_segment import SpeechSegment
from utils.config import Config

# Загружаем переменные окружения
load_dotenv()

# Настраиваем SSL сертификаты
os.environ['SSL_CERT_FILE'] = certifi.where()

logger = logging.getLogger(__name__)

class TranslationLogger:
    """
    Класс для логирования деталей перевода в отдельные файлы
    """
    def __init__(self, log_dir: str = None):
        """
        Инициализация логгера переводов
        
        Args:
            log_dir (str): Директория для хранения логов переводов
        """
        self.session_id = uuid.uuid4().hex[:8]
        self.creation_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем директорию логов если указана или используем временную
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            temp_dir = Path("temp_files")
            self.log_dir = temp_dir / "translation_logs" / f"{self.creation_time}_{self.session_id}"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Файлы для разных типов логов
        self.requests_log_file = self.log_dir / "chatgpt_requests.jsonl"
        self.responses_log_file = self.log_dir / "chatgpt_responses.jsonl"
        self.summary_log_file = self.log_dir / "translation_summary.jsonl"
        
        logger.info(f"Логирование переводов настроено в директории: {self.log_dir}")
    
    def set_log_dir(self, log_dir: str):
        """
        Устанавливает директорию для логов
        
        Args:
            log_dir (str): Путь к директории логов
        """
        log_dir_path = Path(log_dir)
        translation_log_dir = log_dir_path / "translation_logs" / f"{self.creation_time}_{self.session_id}"
        translation_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = translation_log_dir
        self.requests_log_file = self.log_dir / "chatgpt_requests.jsonl"
        self.responses_log_file = self.log_dir / "chatgpt_responses.jsonl"
        self.summary_log_file = self.log_dir / "translation_summary.jsonl"
        
        logger.info(f"Обновлена директория логирования переводов: {self.log_dir}")
    
    def log_request(self, request_id: str, text: str, prompt: str, full_request: Dict[str, Any]):
        """
        Логирует запрос к ChatGPT
        
        Args:
            request_id (str): Уникальный идентификатор запроса
            text (str): Исходный текст для перевода
            prompt (str): Системный промпт
            full_request (Dict): Полный запрос к API
        """
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "original_text": text,
            "system_prompt": prompt,
            "full_request": full_request
        }
        
        with open(self.requests_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def log_response(self, request_id: str, text: str, translated_text: str, 
                     full_response: Dict[str, Any], duration_ms: int):
        """
        Логирует ответ от ChatGPT
        
        Args:
            request_id (str): Уникальный идентификатор запроса
            text (str): Исходный текст для перевода
            translated_text (str): Переведенный текст
            full_response (Dict): Полный ответ от API
            duration_ms (int): Длительность запроса в миллисекундах
        """
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "original_text": text,
            "translated_text": translated_text,
            "duration_ms": duration_ms,
            "full_response": full_response
        }
        
        with open(self.responses_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        # Также добавляем запись в сводный лог
        summary_entry = {
            "request_id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "original_text": text,
            "translated_text": translated_text,
            "duration_ms": duration_ms,
            "success": translated_text is not None and len(translated_text) > 0
        }
        
        with open(self.summary_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

class Translator:
    """
    Класс для перевода текста с поддержкой различных методов перевода
    """
    
    def __init__(self):
        """Инициализация переводчика"""
        self.config = Config()
        self.client = AsyncOpenAI(api_key=self.config.openai_api_key)
        
    async def translate(self, text: str, source_lang: str = "en", target_lang: str = "kk") -> str:
        """
        Переводит отдельный сегмент текста
        
        Args:
            text (str): Текст для перевода
            source_lang (str): Исходный язык
            target_lang (str): Целевой язык
            
        Returns:
            str: Переведенный текст
        """
        try:
            # Используем GPT-4 для перевода
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a professional translator from {source_lang} to {target_lang}. "
                                                "Translate the text keeping its meaning and style. "
                                                "Return ONLY the translation, without any explanations."},
                    {"role": "user", "content": text}
                ]
            )
            
            translated_text = response.choices[0].message.content.strip()
            logger.debug(f"Переведен текст: {text} -> {translated_text}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Ошибка при переводе текста: {str(e)}")
            raise
            
    async def translate_full_context(self, text: str, source_lang: str = "en", target_lang: str = "kk") -> str:
        """
        Переводит весь текст целиком с учетом полного контекста
        
        Args:
            text (str): Полный текст для перевода
            source_lang (str): Исходный язык
            target_lang (str): Целевой язык
            
        Returns:
            str: Переведенный текст
        """
        try:
            # Используем GPT-4 для перевода с полным контекстом
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a professional translator from {source_lang} to {target_lang}. "
                                                "This is a full video transcript. Translate it keeping the overall context, "
                                                "style, and consistency throughout the entire text. Pay special attention to: "
                                                "1. Consistent terminology throughout the text "
                                                "2. Character names and references "
                                                "3. Maintaining the tone and style "
                                                "4. Preserving cultural context where appropriate "
                                                "Return ONLY the translation, without any explanations."},
                    {"role": "user", "content": text}
                ]
            )
            
            translated_text = response.choices[0].message.content.strip()
            logger.info(f"Переведен полный текст длиной {len(text)} символов")
            return translated_text
            
        except Exception as e:
            logger.error(f"Ошибка при переводе полного текста: {str(e)}")
            raise
            
    async def align_translation_with_timings(
        self,
        original_segments: List[Dict[str, Any]],
        translated_full_text: str
    ) -> List[SpeechSegment]:
        """
        Разбивает переведенный текст на сегменты, сохраняя тайминги оригинала
        
        Args:
            original_segments (List[Dict[str, Any]]): Оригинальные сегменты с таймингами
            translated_full_text (str): Полный переведенный текст
            
        Returns:
            List[SpeechSegment]: Список переведенных сегментов с таймингами
        """
        try:
            # Подготавливаем контекст для GPT-4
            segments_info = "\n".join([
                f"{i+1}. [{s['start']:.2f}-{s['end']:.2f}]: {s['text']}"
                for i, s in enumerate(original_segments)
            ])
            
            prompt = f"""Here is the original text with timings:

{segments_info}

And here is the full translation:

{translated_full_text}

Split the translation into segments matching the original timings. Consider:
1. Each segment should fit its time window
2. Sentences should not be cut mid-way if possible
3. The meaning should be preserved
4. The text should sound natural when spoken

Return the result in this exact format, one segment per line:
[start_time-end_time] translated_text

Example:
[0.00-2.50] Translated text for first segment
[2.50-5.00] Translated text for second segment
"""
            
            # Используем GPT-4 для разбивки текста
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional text segmentation assistant. "
                                                "Split the translated text into segments that match the original timings."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Парсим ответ
            segmented_text = response.choices[0].message.content.strip()
            
            # Преобразуем текст в сегменты
            result_segments = []
            for line in segmented_text.split("\n"):
                if not line.strip():
                    continue
                    
                # Извлекаем тайминги и текст
                timing_match = re.match(r"\[(\d+\.\d+)-(\d+\.\d+)\]\s*(.*)", line)
                if timing_match:
                    start_time = float(timing_match.group(1))
                    end_time = float(timing_match.group(2))
                    text = timing_match.group(3).strip()
                    
                    # Создаем сегмент
                    segment = SpeechSegment(
                        text=text,  # Оригинальный текст
                        start_time=start_time,
                        end_time=end_time,
                        speaker_gender='unknown',  # Или берем из оригинального сегмента если есть
                        translated_text=text,  # Переведенный текст
                        audio_path=None  # Аудио будет добавлено позже
                    )
                    result_segments.append(segment)
            
            logger.info(f"Текст успешно разбит на {len(result_segments)} сегментов")
            return result_segments
            
        except Exception as e:
            logger.error(f"Ошибка при разбивке текста на сегменты: {str(e)}")
            raise 