"""
Модуль для высококачественной синхронизации субтитров с учетом длительности речи
и автоматической корректировки временных интервалов с использованием смысловой коррекции через ChatGPT
"""
import os
import logging
import json
import re
import time
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict, field
import numpy as np
from pydub import AudioSegment
from models.speech_segment import SpeechSegment
import asyncio
import openai
import traceback
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class SubtitleTimingData:
    """Класс для хранения данных о временных характеристиках субтитров"""
    start_time: float  # Время начала в секундах
    end_time: float    # Время окончания в секундах
    duration: float    # Длительность в секундах
    text: str         # Текст субтитра
    char_rate: float  # Скорость произношения (символов в секунду)

@dataclass
class SubtitleSegmentationRequest:
    """Класс для представления запроса на сегментацию субтитров"""
    original_text: str
    segments: List[Dict[str, Union[float, str]]]
    target_language: str
    context: str = ""
    
    def to_dict(self):
        """Преобразование в словарь для отправки в API"""
        return {
            "original_text": self.original_text,
            "segments": self.segments,
            "target_language": self.target_language,
            "context": self.context
        }

@dataclass
class SubtitleSegmentationResponse:
    """Класс для представления ответа на запрос сегментации субтитров"""
    segments: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data):
        """Создание объекта из словаря, полученного от API"""
        segments = data.get("segments", [])
        explanations = data.get("explanations", [])
        return cls(segments=segments, explanations=explanations)

class SubtitleSynchronizer:
    """
    Класс для высококачественной синхронизации субтитров с учетом
    длительности речи и автоматической корректировки временных интервалов
    """
    def __init__(self, min_subtitle_duration: float = 1.0, 
                 max_subtitle_duration: float = 7.0,
                 min_gap: float = 0.1,
                 target_char_rate: Dict[str, Tuple[float, float]] = None,
                 openai_api_key: str = None):
        """
        Инициализация синхронизатора субтитров

        Args:
            min_subtitle_duration (float): Минимальная длительность субтитра в секундах
            max_subtitle_duration (float): Максимальная длительность субтитра в секундах
            min_gap (float): Минимальный промежуток между субтитрами в секундах
            target_char_rate (Dict[str, Tuple[float, float]]): Целевая скорость произношения
                для разных языков (символов в секунду). По умолчанию:
                {'en': (10, 15), 'kk': (8, 13)}
            openai_api_key (str): API ключ для доступа к OpenAI API
        """
        self.min_duration = min_subtitle_duration
        self.max_duration = max_subtitle_duration
        self.min_gap = min_gap
        self.target_char_rate = target_char_rate or {
            'en': (10, 15),  # Английский: 10-15 символов в секунду
            'kk': (8, 13)    # Казахский: 8-13 символов в секунду
        }
        
        # Инициализация клиента OpenAI
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        logger.info("Инициализирован синхронизатор субтитров с высоким качеством")

    def _calculate_optimal_duration(self, text: str, lang: str) -> float:
        """
        Рассчитывает оптимальную длительность субтитра на основе текста и языка

        Args:
            text (str): Текст субтитра
            lang (str): Код языка (en/kk)

        Returns:
            float: Оптимальная длительность в секундах
        """
        # Убираем пробелы для точного подсчета символов
        char_count = len(text.replace(' ', ''))
        
        # Получаем целевой диапазон скорости для языка
        min_rate, max_rate = self.target_char_rate.get(lang, (8, 13))
        
        # Рассчитываем оптимальную длительность
        min_duration = char_count / max_rate
        max_duration = char_count / min_rate
        
        # Ограничиваем длительность заданными пределами
        optimal_duration = np.clip(
            (min_duration + max_duration) / 2,
            self.min_duration,
            self.max_duration
        )
        
        return optimal_duration

    def _adjust_timing_for_segment(self, 
                                 segment: SpeechSegment,
                                 prev_end: float = 0.0) -> Tuple[float, float]:
        """
        Корректирует временные метки для сегмента с учетом предыдущего сегмента

        Args:
            segment (SpeechSegment): Сегмент для корректировки
            prev_end (float): Время окончания предыдущего сегмента

        Returns:
            Tuple[float, float]: Новые время начала и окончания
        """
        # Рассчитываем оптимальную длительность для переведенного текста
        optimal_duration = self._calculate_optimal_duration(
            segment.translated_text, 'kk'
        )
        
        # Обеспечиваем минимальный промежуток после предыдущего субтитра
        new_start = max(segment.start_time, prev_end + self.min_gap)
        
        # Рассчитываем новое время окончания
        new_end = new_start + optimal_duration
        
        # Если новый конец выходит за пределы следующего сегмента,
        # корректируем длительность
        if new_end > segment.end_time:
            overlap = new_end - segment.end_time
            if overlap < optimal_duration * 0.3:  # Если перекрытие небольшое
                new_end = segment.end_time
            else:
                # Пропорционально сжимаем длительность
                new_duration = min(
                    optimal_duration,
                    segment.end_time - new_start
                )
                new_end = new_start + new_duration
        
        return new_start, new_end

    async def _optimize_segment_with_gpt(self, text: str, max_chars: int, target_language: str) -> str:
        """
        Оптимизирует текст с использованием GPT для обеспечения смыслового соответствия 
        и заданной длины

        Args:
            text (str): Исходный текст
            max_chars (int): Максимальное количество символов
            target_language (str): Целевой язык

        Returns:
            str: Оптимизированный текст
        """
        try:
            if not self.openai_api_key:
                logger.warning("API ключ OpenAI не указан, оптимизация текста невозможна")
                return text
            
            # Создаём системный промпт для GPT-4o
            system_prompt = f"""Ты эксперт по оптимизации текста для субтитров.
Твоя задача - оптимизировать переведенный текст для субтитров, сохраняя его смысл,
но адаптируя его длину для лучшей читаемости и синхронизации с видео.
Текст должен сохранять все важные детали и быть максимально близким к оригиналу.
Target language: {target_language}
Maximum characters: {max_chars}"""

            user_prompt = f"""Оптимизируй следующий текст для субтитров, сохраняя смысл,
но не превышая {max_chars} символов. Текст должен звучать естественно на языке {target_language}:

"{text}"

Верни только оптимизированный текст, без дополнительных пояснений."""

            # Создаем запрос к API
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Получаем ответ от GPT
            optimized_text = response.choices[0].message.content.strip()
            
            # Удаляем кавычки, если они есть в ответе
            optimized_text = re.sub(r'^["\']|["\']$', '', optimized_text)
            
            logger.info(f"Текст оптимизирован с помощью GPT: '{text}' -> '{optimized_text}'")
            logger.info(f"Исходная длина: {len(text)}, оптимизированная длина: {len(optimized_text)} символов")
            return optimized_text
            
        except Exception as e:
            logger.error(f"Ошибка при оптимизации текста с GPT: {str(e)}")
            logger.error(traceback.format_exc())
            return text

    async def _segment_by_meaning(self, segments: List[SpeechSegment], target_language: str) -> List[SpeechSegment]:
        """
        Сегментирует субтитры по смыслу с использованием GPT

        Args:
            segments (List[SpeechSegment]): Исходные сегменты
            target_language (str): Целевой язык

        Returns:
            List[SpeechSegment]: Сегменты с оптимизированными границами
        """
        try:
            if not self.openai_api_key:
                logger.warning("API ключ OpenAI не указан, сегментация по смыслу невозможна")
                return segments
            
            if len(segments) <= 1:
                return segments
                
            # Получаем полный текст для контекста
            full_text = " ".join([s.text for s in segments])
            
            # Подготавливаем данные для запроса
            segments_data = []
            for i, segment in enumerate(segments):
                segments_data.append({
                    "id": i,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                    "translated_text": segment.translated_text if hasattr(segment, "translated_text") else ""
                })
            
            # Создаем запрос
            request = SubtitleSegmentationRequest(
                original_text=full_text,
                segments=segments_data,
                target_language=target_language
            )
            
            # Создаём системный промпт для GPT-4o
            system_prompt = f"""Ты эксперт по сегментации субтитров для перевода с английского на {target_language}.
Твоя задача - проанализировать сегменты субтитров и предложить улучшенное разделение текста,
основываясь на смысловых единицах, чтобы обеспечить наилучшее качество перевода и синхронизации.
Ты можешь:
1. Изменить границы сегментов (объединить короткие связанные сегменты или разделить длинные)
2. Рекомендовать оптимальное время начала и окончания для каждого сегмента
3. Учитывать смысловую завершенность фраз при сегментации

Твои рекомендации должны быть основаны на лингвистическом анализе и учитывать особенности целевого языка ({target_language}).
"""

            # Создаем пользовательский промпт с данными
            user_prompt = f"""Проанализируй следующие сегменты субтитров и предложи улучшенное смысловое разделение:

Полный текст: "{full_text}"

Текущие сегменты:
"""
            for segment in segments_data:
                user_prompt += f"ID: {segment['id']}, Время: [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s], Текст: \"{segment['text']}\"\n"
            
            user_prompt += f"""
Предложи оптимальное смысловое разделение этих субтитров для перевода на {target_language}.
Ты можешь предложить новые границы сегментов, объединить или разделить их для улучшения качества перевода.

Верни ответ в формате JSON:
{{
  "segments": [
    {{
      "id": 0,
      "start_time": 1.2,
      "end_time": 3.5,
      "original_text": "Текст на английском",
      "notes": "Причина, по которой этот сегмент оптимален"
    }},
    ...
  ],
  "explanations": ["Общее пояснение к изменениям"]
}}
"""

            # Создаем запрос к API
            client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info("Отправляем запрос для сегментации по смыслу в GPT-4o...")
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Получаем ответ от GPT
            try:
                response_text = response.choices[0].message.content.strip()
                logger.info(f"Получен ответ от GPT по сегментации субтитров: {response_text}")
                response_data = json.loads(response_text)
                logger.info(f"Получен ответ от GPT по сегментации субтитров")
                
                # Создаем новые сегменты на основе рекомендаций GPT
                optimized_segments = []
                
                # Для сохранения соответствия между старыми и новыми сегментами
                segment_map = {}
                for old_idx, old_segment in enumerate(segments):
                    segment_map[old_idx] = old_segment
                
                # Выводим подробную информацию о новой сегментации
                logger.info("--- РЕЗУЛЬТАТЫ СМЫСЛОВОЙ СЕГМЕНТАЦИИ ---")
                
                # Создаем новые сегменты
                for new_segment_data in response_data.get("segments", []):
                    # Получаем данные из GPT
                    start_time = float(new_segment_data.get("start_time", 0))
                    end_time = float(new_segment_data.get("end_time", 0))
                    original_text = new_segment_data.get("original_text", "")
                    notes = new_segment_data.get("notes", "")
                    
                    # Выводим информацию о новом сегменте
                    logger.info(f"Новый сегмент: [{start_time:.2f}s - {end_time:.2f}s] '{original_text}'")
                    if notes:
                        logger.info(f"Примечание: {notes}")
                    
                    # Создаем новый сегмент
                    new_segment = SpeechSegment(
                        start_time=start_time,
                        end_time=end_time,
                        text=original_text
                    )
                    
                    # Переносим свойства из старых сегментов, если можем найти соответствие
                    # Находим наиболее перекрывающийся старый сегмент
                    best_overlap = 0
                    best_match = None
                    
                    for old_segment in segments:
                        overlap_start = max(old_segment.start_time, start_time)
                        overlap_end = min(old_segment.end_time, end_time)
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = old_segment
                    
                    # Переносим свойства из лучшего совпадения, если оно есть
                    if best_match:
                        # Сохраняем пол говорящего из лучшего совпадения
                        # Важно: это критично для правильного выбора голоса при синтезе речи
                        new_segment.speaker_gender = best_match.speaker_gender
                        logger.debug(f"Перенесен пол говорящего: {new_segment.speaker_gender}")
                        
                        # Переносим путь к аудио, если есть
                        new_segment.audio_path = best_match.audio_path if hasattr(best_match, "audio_path") else None
                        
                        # Переносим перевод, если есть
                        if hasattr(best_match, "translated_text") and best_match.translated_text:
                            new_segment.translated_text = best_match.translated_text
                            logger.info(f"Перенесенный перевод: '{new_segment.translated_text}'")
                            
                        # Сохраняем уверенность в определении пола, если есть
                        if hasattr(best_match, "gender_confidence"):
                            new_segment.gender_confidence = best_match.gender_confidence
                            
                    else:
                        # Если нет наилучшего совпадения, пробуем найти похожие сегменты по времени
                        # для определения пола говорящего
                        overlapping_segments = []
                        start_time = new_segment.start_time
                        end_time = new_segment.end_time
                        
                        for old_segment in segments:
                            # Проверяем, если сегменты перекрываются
                            if (old_segment.start_time <= end_time and 
                                old_segment.end_time >= start_time):
                                overlapping_segments.append(old_segment)
                        
                        # Используем самый часто встречающийся пол в перекрывающихся сегментах
                        if overlapping_segments:
                            # Собираем данные о поле говорящих
                            genders = [s.speaker_gender for s in overlapping_segments 
                                      if hasattr(s, 'speaker_gender') and s.speaker_gender != 'unknown']
                            
                            # Определяем наиболее часто встречающийся пол
                            if genders:
                                from collections import Counter
                                most_common_gender = Counter(genders).most_common(1)[0][0]
                                new_segment.speaker_gender = most_common_gender
                                logger.debug(f"Определен пол говорящего на основе перекрытий: {most_common_gender}")
                    
                    optimized_segments.append(new_segment)
                
                # Логируем объяснения от GPT
                logger.info("--- ОБЪЯСНЕНИЯ GPT ---")
                for explanation in response_data.get("explanations", []):
                    logger.info(f"GPT объяснение: {explanation}")
                
                logger.info(f"Оптимизировано {len(segments)} -> {len(optimized_segments)} сегментов")
                return optimized_segments
                
            except json.JSONDecodeError:
                logger.error(f"Ошибка при разборе JSON ответа от GPT")
                logger.error(f"Полученный ответ: {response.choices[0].message.content}")
                return segments
            
        except Exception as e:
            logger.error(f"Ошибка при сегментации по смыслу: {str(e)}")
            logger.error(traceback.format_exc())
            return segments

    async def _adapt_translation_for_timing(self, 
                                         segment: SpeechSegment, 
                                         target_language: str) -> SpeechSegment:
        """
        Адаптирует перевод под временные ограничения с помощью GPT

        Args:
            segment (SpeechSegment): Сегмент для адаптации
            target_language (str): Целевой язык перевода

        Returns:
            SpeechSegment: Сегмент с адаптированным переводом
        """
        if not hasattr(segment, "translated_text") or not segment.translated_text:
            return segment
            
        # Рассчитываем максимальное количество символов
        duration = segment.end_time - segment.start_time
        max_chars = int(duration * self.target_char_rate.get(target_language, (8, 13))[1])
        
        # Если перевод укладывается в ограничения, ничего не меняем
        if len(segment.translated_text) <= max_chars:
            return segment
            
        try:
            # Оптимизируем перевод
            optimized_translation = await self._optimize_segment_with_gpt(
                segment.translated_text, 
                max_chars,
                target_language
            )
            
            segment.translated_text = optimized_translation
            return segment
            
        except Exception as e:
            logger.error(f"Ошибка при адаптации перевода: {str(e)}")
            return segment

    async def synchronize_segments_with_gpt(self, 
                                         segments: List[SpeechSegment],
                                         target_language: str = 'kk',
                                         high_quality: bool = True) -> List[SpeechSegment]:
        """
        Синхронизирует сегменты субтитров с учетом смысла и длительности с помощью GPT

        Args:
            segments (List[SpeechSegment]): Список сегментов для синхронизации
            target_language (str): Целевой язык перевода
            high_quality (bool): Использовать ли режим высокого качества

        Returns:
            List[SpeechSegment]: Список синхронизированных сегментов
        """
        if not segments:
            return segments

        logger.info(f"Начинаем смысловую синхронизацию {len(segments)} сегментов "
                   f"(режим высокого качества: {high_quality})")

        try:
            # Показываем примеры исходных сегментов
            logger.info("Примеры исходных сегментов:")
            for i, segment in enumerate(segments[:5]):  # Показываем первые 5 сегментов
                logger.info(f"  Сегмент {i+1}: [{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                           f"Текст: '{segment.text[:50]}{'...' if len(segment.text) > 50 else ''}', "
                           f"Перевод: '{segment.translated_text[:50] if hasattr(segment, 'translated_text') else 'нет'}{'...' if hasattr(segment, 'translated_text') and len(segment.translated_text) > 50 else ''}'")
            
            # Шаг 1: Сегментация по смыслу
            if high_quality and len(segments) > 1:
                logger.info("Шаг 1: Применяем сегментацию по смыслу...")
                optimized_segments = await self._segment_by_meaning(segments, target_language)
                logger.info(f"Сегментация по смыслу завершена. Получено {len(optimized_segments)} сегментов.")
            else:
                optimized_segments = segments.copy()
                logger.info("Сегментация по смыслу пропущена (использованы исходные сегменты).")
            
            # Шаг 2: Адаптация переводов для каждого сегмента
            logger.info("Шаг 2: Адаптация переводов под временные ограничения...")
            for i, segment in enumerate(optimized_segments):
                if hasattr(segment, "translated_text") and segment.translated_text:
                    logger.info(f"Адаптация перевода для сегмента {i+1}/{len(optimized_segments)}: "
                               f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                               f"'{segment.translated_text[:50]}{'...' if len(segment.translated_text) > 50 else ''}'")
                    segment = await self._adapt_translation_for_timing(segment, target_language)
                    optimized_segments[i] = segment
            
            # Шаг 3: Техническая синхронизация с учетом адаптированных переводов
            logger.info("Шаг 3: Техническая синхронизация тайминга...")
            synced_segments = self.synchronize_segments(optimized_segments, high_quality)
            
            # Выводим примеры синхронизированных сегментов
            logger.info("Примеры синхронизированных сегментов:")
            for i, segment in enumerate(synced_segments[:5]):  # Показываем первые 5 сегментов
                logger.info(f"  Сегмент {i+1}: [{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                           f"Длительность: {segment.end_time - segment.start_time:.2f}s, "
                           f"Перевод: '{segment.translated_text[:50] if hasattr(segment, 'translated_text') else 'нет'}{'...' if hasattr(segment, 'translated_text') and len(segment.translated_text) > 50 else ''}'")
            
            logger.info(f"Смысловая синхронизация завершена, обработано {len(synced_segments)} сегментов")
            return synced_segments
            
        except Exception as e:
            logger.error(f"Ошибка при смысловой синхронизации: {str(e)}")
            logger.error(traceback.format_exc())
            # В случае ошибки возвращаем исходные сегменты после базовой синхронизации
            return self.synchronize_segments(segments, high_quality)

    def synchronize_segments(self, 
                           segments: List[SpeechSegment],
                           high_quality: bool = True) -> List[SpeechSegment]:
        """
        Синхронизирует сегменты субтитров с учетом длительности речи,
        сохраняя все важные атрибуты, включая пол говорящего

        Args:
            segments (List[SpeechSegment]): Список сегментов для синхронизации
            high_quality (bool): Использовать ли режим высокого качества

        Returns:
            List[SpeechSegment]: Список синхронизированных сегментов
        """
        if not segments:
            return segments

        logger.info(f"Начинаем синхронизацию {len(segments)} сегментов "
                   f"(режим высокого качества: {high_quality})")

        # Важно: логгируем полы говорящих в исходных сегментах для отладки
        logger.debug("Полы говорящих в исходных сегментах:")
        for i, segment in enumerate(segments[:10]):  # Показываем для первых 10 сегментов
            logger.debug(f"  Сегмент {i+1}: [{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                       f"Пол говорящего: {segment.speaker_gender}")
        
        # Создаем глубокие копии сегментов для обработки, сохраняя все атрибуты
        synced_segments = deepcopy(segments)
        
        if high_quality:
            # Первый проход: анализ и оптимизация длительностей
            timing_data = []
            for segment in synced_segments:
                optimal_duration = self._calculate_optimal_duration(
                    segment.translated_text if hasattr(segment, "translated_text") and segment.translated_text else segment.text, 
                    'kk'
                )
                timing_data.append(SubtitleTimingData(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    duration=optimal_duration,
                    text=segment.translated_text if hasattr(segment, "translated_text") and segment.translated_text else segment.text,
                    char_rate=len((segment.translated_text if hasattr(segment, "translated_text") and segment.translated_text else segment.text).replace(' ', '')) / optimal_duration
                ))

            # Второй проход: глобальная оптимизация
            prev_end = 0.0
            for i, segment in enumerate(synced_segments):
                # Получаем оптимальные временные метки
                new_start, new_end = self._adjust_timing_for_segment(segment, prev_end)
                
                # Применяем корректировку с учетом следующего сегмента
                if i < len(synced_segments) - 1:
                    next_segment = synced_segments[i + 1]
                    if new_end + self.min_gap > next_segment.start_time:
                        # Корректируем длительность, сохраняя естественность речи
                        available_duration = next_segment.start_time - new_start - self.min_gap
                        if available_duration >= self.min_duration:
                            new_end = new_start + available_duration
                
                # Обновляем временные метки сегмента
                segment.start_time = new_start
                segment.end_time = new_end
                prev_end = new_end
                
                logger.debug(
                    f"Сегмент {i+1}: "
                    f"[{new_start:.2f}s - {new_end:.2f}s] "
                    f"Длительность: {new_end - new_start:.2f}s, "
                    f"Пол говорящего: {segment.speaker_gender}"
                )
        else:
            # Простая синхронизация без глубокой оптимизации
            prev_end = 0.0
            for i, segment in enumerate(synced_segments):
                new_start = max(segment.start_time, prev_end + self.min_gap)
                new_end = max(new_start + self.min_duration, segment.end_time)
                
                # Обновляем временные метки сегмента
                segment.start_time = new_start
                segment.end_time = new_end
                prev_end = new_end
                
                logger.debug(
                    f"Сегмент {i+1}: "
                    f"[{new_start:.2f}s - {new_end:.2f}s] "
                    f"Длительность: {new_end - new_start:.2f}s, "
                    f"Пол говорящего: {segment.speaker_gender}"
                )
        
        # Проверяем и логируем информацию о поле говорящих в синхронизированных сегментах
        logger.debug("Полы говорящих в синхронизированных сегментах:")
        for i, segment in enumerate(synced_segments[:10]):
            logger.debug(f"  Сегмент {i+1}: [{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                       f"Пол говорящего: {segment.speaker_gender}")
        
        logger.info(f"Синхронизация завершена, обработано {len(synced_segments)} сегментов")
        return synced_segments

    def apply_audio_based_adjustments(self,
                                    segments: List[SpeechSegment],
                                    audio_path: str) -> List[SpeechSegment]:
        """
        Применяет дополнительные корректировки на основе анализа аудио

        Args:
            segments (List[SpeechSegment]): Список сегментов для корректировки
            audio_path (str): Путь к аудиофайлу синтезированной речи

        Returns:
            List[SpeechSegment]: Список скорректированных сегментов
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Создаем копию сегментов
            adjusted_segments = segments.copy()
            
            for segment in adjusted_segments:
                if not segment.audio_path or not os.path.exists(segment.audio_path):
                    continue
                
                # Загружаем аудио сегмента
                segment_audio = AudioSegment.from_file(segment.audio_path)
                actual_duration = len(segment_audio) / 1000.0  # в секундах
                
                # Если длительность синтезированного аудио сильно отличается
                # от длительности сегмента, корректируем временные метки
                current_duration = segment.end_time - segment.start_time
                if abs(actual_duration - current_duration) > 0.3:  # Порог в 300 мс
                    # Корректируем длительность с учетом соседних сегментов
                    new_duration = min(
                        max(actual_duration, self.min_duration),
                        self.max_duration
                    )
                    
                    # Пытаемся центрировать новую длительность в доступном окне
                    available_shift = (current_duration - new_duration) / 2
                    if available_shift > 0:
                        segment.start_time += available_shift
                        segment.end_time = segment.start_time + new_duration
                    else:
                        segment.end_time = segment.start_time + new_duration
            
            logger.info("Применены корректировки на основе анализа аудио")
            return adjusted_segments
            
        except Exception as e:
            logger.error(f"Ошибка при корректировке на основе аудио: {e}")
            return segments 