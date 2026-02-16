"""
Модуль для синхронизированного синтеза речи с адаптивным управлением скоростью и качеством
"""
import os
import logging
import asyncio
import tempfile
from pydub import AudioSegment
from typing import List, Optional, Dict, Any, Tuple
import time
import re
import json
import random

import edge_tts
from models.speech_segment import SpeechSegment
from services.tts_manager import TTSManager

logger = logging.getLogger(__name__)

class SynchronizedSynthesizer:
    """
    Класс для синхронизированного синтеза речи
    """
    def __init__(self, tts_manager: TTSManager = None):
        """
        Инициализация синтезатора
        
        Args:
            tts_manager (TTSManager, optional): Менеджер TTS
        """
        self.tts_manager = tts_manager or TTSManager()
        logger.info("Инициализирован синхронизированный синтезатор речи")
    
    async def synthesize_segments(
        self,
        segments: List[SpeechSegment],
        output_dir: str,
        language: str = 'kk',
        voice: str = None
    ) -> List[SpeechSegment]:
        """
        Синтез речи для списка сегментов
        
        Args:
            segments (List[SpeechSegment]): Список сегментов для синтеза
            output_dir (str): Директория для сохранения синтезированных файлов
            language (str): Язык синтеза
            voice (str, optional): Голос для синтеза
            
        Returns:
            List[SpeechSegment]: Список сегментов с путями к синтезированным файлам
        """
        try:
            # Создаем директорию для синтезированных файлов
            os.makedirs(output_dir, exist_ok=True)
            
            # Синтезируем каждый сегмент
            tasks = []
            for i, segment in enumerate(segments):
                try:
                    # Формируем путь для сохранения
                    output_path = os.path.join(output_dir, f"segment_{i+1}.wav")
                    
                    # Создаем задачу для синтеза
                    task = self.tts_manager.synthesize(
                        text=segment.translated_text,
                        output_path=output_path,
                        language=language,
                        voice=voice
                    )
                    tasks.append((i, segment, task))
                    
                except Exception as e:
                    logger.error(f"Ошибка при создании задачи синтеза для сегмента {i+1}: {e}")
            
            # Ожидаем завершения всех задач
            for i, segment, task in tasks:
                try:
                    # Ожидаем результат задачи
                    result = await task
                    
                    if result:
                        # Обновляем путь к аудио в сегменте
                        output_path = os.path.join(output_dir, f"segment_{i+1}.wav")
                        segment.audio_path = output_path
                        logger.info(f"Синтезирован сегмент {i+1}: {segment.translated_text[:50]}...")
                    else:
                        logger.error(f"Не удалось синтезировать сегмент {i+1}")
                        
                except Exception as e:
                    logger.error(f"Ошибка при синтезе сегмента {i+1}: {e}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка при синтезе сегментов: {e}")
            return segments

class SynchronizedSpeechSynthesizer:
    """
    Класс для высококачественного синтеза речи с синхронизацией по времени оригинальных сегментов
    """
    def __init__(self, max_parallel_tasks=5):
        """
        Инициализация синтезатора синхронизированной речи
        
        Args:
            max_parallel_tasks (int): Максимальное количество параллельных задач синтеза
        """
        # Кеш доступных голосов
        self.voices = None
        
        # Карта голосов для разных языков и полов
        self.voice_map = {
            'kk': {  # Казахский
                'male': "kk-KZ-DauletNeural", 
                'female': "kk-KZ-AigulNeural"
            },
            'ru': {  # Русский (для поддержки мультиязычности)
                'male': "ru-RU-DmitryNeural",
                'female': "ru-RU-SvetlanaNeural"
            }
        }
        
        # Ограничение параллельных задач для предотвращения перегрузки
        self.max_parallel_tasks = max_parallel_tasks
        self.semaphore = asyncio.Semaphore(max_parallel_tasks)
        
        # Кэш TTS для одинаковых текстов и голосов
        self.tts_cache = {}
        
        # Коэффициенты для адаптации скорости речи
        self.language_speed_coefficients = {
            'kk': 1.2,  # Казахский произносится медленнее английского
            'ru': 1.1,  # Русский тоже немного медленнее
            'default': 1.0
        }
        
        # Информация о перерывах между словами для каждого языка
        self.pause_patterns = {
            'kk': [',', '.', '!', '?', ':', ';', '-'],
            'ru': [',', '.', '!', '?', ':', ';', '-']
        }
        
        # Настройки качества для разных длин текста
        self.quality_settings = {
            'short': {
                'speed_multiplier': 0.85,  # Замедляем для лучшего восприятия
                'min_speed': 0.5,          # Минимальная скорость
                'max_speed': 1.5           # Максимальная скорость
            },
            'medium': {
                'speed_multiplier': 0.92,
                'min_speed': 0.6,
                'max_speed': 1.7
            },
            'long': {
                'speed_multiplier': 1.0,   # Стандартная скорость для длинных фраз
                'min_speed': 0.7,
                'max_speed': 1.8
            }
        }
        
        # Кэш медленных синтезов для анализа и отладки
        self.slow_synthesis_cache = []
        
        # Настройки для голосов по умолчанию (профили)
        self.voice_profiles = {}
        
        # Каталог для сохранения временных файлов и кэша
        self.temp_dir = os.path.join(tempfile.gettempdir(), "speech_synthesizer_cache")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Загружаем кэш из файла, если существует
        self._load_cache()
        
        logger.info(f"Инициализирован высококачественный синтезатор речи (макс. параллельных задач: {max_parallel_tasks})")
    
    def _load_cache(self):
        """Загружает кэш из файла, если он существует"""
        cache_path = os.path.join(self.temp_dir, "tts_cache.json")
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Загружаем только основные параметры, без аудио-данных
                    for key, value in cached_data.items():
                        if isinstance(value, dict) and 'duration' in value and 'rate' in value:
                            self.tts_cache[key] = value
                logger.info(f"Загружен кэш TTS с {len(self.tts_cache)} записями")
        except Exception as e:
            logger.warning(f"Не удалось загрузить кэш TTS: {e}")
            self.tts_cache = {}
    
    def _save_cache(self):
        """Сохраняет кэш в файл для повторного использования"""
        cache_path = os.path.join(self.temp_dir, "tts_cache.json")
        try:
            # Сохраняем только если в кэше есть данные
            if self.tts_cache:
                # Ограничиваем размер кэша (не более 1000 записей)
                if len(self.tts_cache) > 1000:
                    # Оставляем только 800 самых новых записей
                    keys = list(self.tts_cache.keys())
                    for key in keys[:-800]:
                        del self.tts_cache[key]
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.tts_cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Сохранен кэш TTS с {len(self.tts_cache)} записями")
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш TTS: {e}")
    
    async def _get_voices(self):
        """
        Получает список доступных голосов Edge TTS
        
        Returns:
            list: Список доступных голосов
        """
        try:
            voices = await edge_tts.list_voices()
            
            # Сохраняем профили голосов для дальнейшего использования
            for voice in voices:
                voice_name = voice.get("ShortName")
                if voice_name:
                    self.voice_profiles[voice_name] = {
                        "gender": voice.get("Gender"),
                        "locale": voice.get("Locale"),
                        "name": voice.get("Name")
                    }
            
            return voices
        except Exception as e:
            logger.error(f"Ошибка при получении списка голосов: {e}")
            return []
    
    def _ensure_voices_loaded(self, loop=None):
        """Загрузка голосов, если они еще не загружены"""
        if self.voices is None:
            try:
                # Получаем или создаем цикл событий
                loop = loop or asyncio.get_event_loop()
                if loop.is_running():
                    # Используем новый цикл, если текущий занят
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Загружаем голоса
                self.voices = loop.run_until_complete(self._get_voices())
                
                # Если голоса загружены успешно, обновляем карту голосов
                if self.voices:
                    # Заполняем голоса для казахского языка
                    kk_voices = [v for v in self.voices if v.get("Locale", "").startswith("kk")]
                    if kk_voices:
                        for voice in kk_voices:
                            gender = voice.get("Gender", "")
                            if "Male" in gender and voice.get("ShortName"):
                                self.voice_map['kk']['male'] = voice.get("ShortName")
                            elif "Female" in gender and voice.get("ShortName"):
                                self.voice_map['kk']['female'] = voice.get("ShortName")
                
                    # Заполняем голоса для русского языка
                    ru_voices = [v for v in self.voices if v.get("Locale", "").startswith("ru")]
                    if ru_voices:
                        for voice in ru_voices:
                            gender = voice.get("Gender", "")
                            if "Male" in gender and voice.get("ShortName"):
                                self.voice_map['ru']['male'] = voice.get("ShortName")
                            elif "Female" in gender and voice.get("ShortName"):
                                self.voice_map['ru']['female'] = voice.get("ShortName")
                
                logger.info(f"Загружено {len(self.voices)} голосов для синтеза речи")
            except Exception as e:
                logger.error(f"Ошибка при загрузке голосов: {e}")
                # В случае ошибки инициализируем пустым списком
                self.voices = []
    
    def _select_voice(self, gender, language='kk', loop=None):
        """
        Умное чередование голосов для естественного звучания диалогов, 
        не полагаясь на определение пола YAMNet
        
        Args:
            gender (str): Пол говорящего ('male', 'female', 'unknown') - используется как подсказка, но не обязателен
            language (str): Код языка (например, 'kk' для казахского)
            loop (asyncio.AbstractEventLoop, optional): Цикл событий
            
        Returns:
            str: Идентификатор голоса
        """
        try:
            # Обеспечиваем загрузку голосов
            self._ensure_voices_loaded(loop)
            
            # Если язык не поддерживается явно, используем казахский
            if language not in self.voice_map:
                logger.warning(f"Язык {language} не найден в карте голосов, используем казахский")
                language = 'kk'
                
            # Определяем, нужно ли использовать умное чередование голосов
            use_smart_alternation = True  # По умолчанию используем умное чередование
            
            # В первый раз инициализируем последний выбранный голос и счетчик последовательности
            if not hasattr(self, '_last_selected_gender'):
                self._last_selected_gender = None
                self._consecutive_same_gender_count = 0
                self._segment_counter = 0
                
            # Инкрементируем счетчик сегментов для управления чередованием
            self._segment_counter += 1
            
            # Интеллектуальный алгоритм выбора пола говорящего:
            selected_gender = gender  # Исходный пол как предложение
                
            # Правила интеллектуального чередования:
            if use_smart_alternation:
                # 1. Предотвращаем длинные последовательности одного и того же пола
                if self._last_selected_gender == gender:
                    self._consecutive_same_gender_count += 1
                    # Если уже было 3-4 сегмента подряд одного пола, увеличиваем вероятность смены
                    if self._consecutive_same_gender_count >= 3:
                        change_prob = 0.7  # 70% вероятность сменить пол
                        if random.random() < change_prob:
                            selected_gender = 'female' if gender != 'female' else 'male'
                            logger.info(f"Принудительная смена пола после {self._consecutive_same_gender_count} последовательных сегментов")
                else:
                    # Сбрасываем счетчик при смене пола
                    self._consecutive_same_gender_count = 0
                    
                # 2. Для неизвестного пола используем интеллектуальное чередование
                if gender == 'unknown' or gender not in ('male', 'female'):
                    # Чередуем мужской/женский для неизвестного пола с учетом предыдущей последовательности
                    if self._last_selected_gender == 'male':
                        selected_gender = 'female'  # Если предыдущий был мужской, выбираем женский
                    elif self._last_selected_gender == 'female':
                        selected_gender = 'male'    # Если предыдущий был женский, выбираем мужской
                    else:
                        # Если нет предыдущего или он неизвестен, используем позицию в диалоге
                        # Нечетные позиции - мужской голос, четные - женский
                        selected_gender = 'male' if self._segment_counter % 2 == 1 else 'female'
                    
                # 3. Для коротких диалогов с двумя говорящими стараемся чередовать мужской и женский
                if self._segment_counter <= 10:  # Для первых 10 сегментов
                    # Применяем простое правило: первый, третий, пятый и т.д. - мужской
                    if self._segment_counter <= 2:  # Для первых двух сегментов жестко задаем разные голоса
                        selected_gender = 'male' if self._segment_counter == 1 else 'female'
            
            # Обновляем последний выбранный пол
            self._last_selected_gender = selected_gender
            
            # Выбираем голос из карты в соответствии с выбранным полом
            voice = self.voice_map[language][selected_gender]
            
            # Проверяем, существует ли выбранный голос в загруженном списке
            if self.voices and not any(v.get("ShortName") == voice for v in self.voices):
                logger.warning(f"Голос {voice} не найден в списке доступных голосов, ищем альтернативу")
                
                # Если голос не найден, ищем любой подходящий для этого языка
                target_voices = [v for v in self.voices if v.get("Locale", "").startswith(language)]
                
                if target_voices:
                    if selected_gender == 'male':
                        male_voices = [v for v in target_voices if "Male" in v.get("Gender", "")]
                        if male_voices:
                            voice = male_voices[0]["ShortName"]
                            logger.info(f"Найден альтернативный мужской голос: {voice}")
                    elif selected_gender == 'female':
                        female_voices = [v for v in target_voices if "Female" in v.get("Gender", "")]
                        if female_voices:
                            voice = female_voices[0]["ShortName"]
                            logger.info(f"Найден альтернативный женский голос: {voice}")
            
            logger.info(f"Выбран голос {voice} для пола {selected_gender} и языка {language} (исходный пол: {gender})")
            return voice
            
        except Exception as e:
            logger.error(f"Ошибка при выборе голоса: {e}")
            # Возвращаем голос по умолчанию для указанного языка
            default_gender = 'male' if random.random() < 0.5 else 'female'
            return self.voice_map.get(language, self.voice_map['kk'])[default_gender]
    
    def _get_text_characteristics(self, text, language='kk'):
        """
        Анализирует характеристики текста для оптимального синтеза
        
        Args:
            text (str): Текст для анализа
            language (str): Код языка текста
            
        Returns:
            dict: Характеристики текста (длина, категория, паузы)
        """
        # Определяем длину текста
        text_length = len(text)
        
        # Определяем категорию длины текста
        if text_length < 10:
            length_category = 'short'
        elif text_length < 30:
            length_category = 'medium'
        else:
            length_category = 'long'
        
        # Считаем количество пауз (знаки препинания)
        pause_count = 0
        if language in self.pause_patterns:
            for pause_char in self.pause_patterns[language]:
                pause_count += text.count(pause_char)
        
        # Считаем количество слов
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # Определяем вероятность наличия эмоциональной речи
        emotional = '!' in text or '?' in text
        
        return {
            'length': text_length,
            'category': length_category,
            'pause_count': pause_count,
            'word_count': word_count,
            'emotional': emotional
        }
    
    def _calculate_optimal_speed(self, original_duration, synthesized_duration, text, language='kk'):
        """
        Рассчитывает оптимальную скорость для синтеза речи с учетом характеристик текста
        
        Args:
            original_duration (float): Длительность оригинального сегмента в секундах
            synthesized_duration (float): Длительность синтезированной речи в секундах
            text (str): Текст для синтеза
            language (str): Код языка
            
        Returns:
            str: Строка с оптимальной скоростью для Edge TTS
        """
        # Получаем характеристики текста
        characteristics = self._get_text_characteristics(text, language)
        text_length = characteristics['length']
        category = characteristics['category']
        pause_count = characteristics['pause_count']
        word_count = characteristics['word_count']
        emotional = characteristics['emotional']
        
        # Базовый коэффициент скорости
        raw_ratio = synthesized_duration / original_duration
        
        # Получаем настройки качества для данной категории текста
        quality = self.quality_settings[category]
        
        # Корректируем соотношение с учетом длины и пауз
        pause_factor = 1.0
        if pause_count > 0:
            # Больше пауз = больше времени нужно на произношение
            pause_factor = 1.0 - (min(pause_count, 5) * 0.03)  # До 15% замедления при >5 паузах
        
        # Учитываем среднюю длину слов (для языков с длинными словами нужно больше времени)
        word_length_factor = 1.0
        if word_count > 0:
            avg_word_length = text_length / word_count
            if avg_word_length > 7:  # Длинные слова
                word_length_factor = 0.95  # Замедление на 5%
        
        # Учитываем эмоциональную речь
        emotion_factor = 0.95 if emotional else 1.0  # Эмоциональную речь произносим чуть медленнее
        
        # Учитываем специфику целевого языка
        language_factor = self.language_speed_coefficients.get(language, 
                                                               self.language_speed_coefficients['default'])
        
        # Собираем все факторы вместе
        adjusted_ratio = raw_ratio * pause_factor * word_length_factor * emotion_factor / language_factor
        
        # Применяем множитель скорости для данной категории текста
        adjusted_ratio *= quality['speed_multiplier']
        
        # Ограничиваем значения для предотвращения слишком быстрой или медленной речи
        target_ratio = max(min(1.0 / adjusted_ratio, quality['max_speed']), quality['min_speed'])
        
        # Преобразуем в процентное изменение для Edge TTS
        if target_ratio < 1.0:
            # Замедление (отрицательный процент)
            rate = f"-{int((1.0 - target_ratio) * 100)}%"
        else:
            # Ускорение (положительный процент)
            rate = f"+{int((target_ratio - 1.0) * 100)}%"
        
        # Подробный лог для отладки
        logger.debug(
            f"Расчет скорости: оригинал {original_duration:.2f}с, синтез {synthesized_duration:.2f}с, "
            f"длина {text_length}, слов {word_count}, пауз {pause_count}, "
            f"исходный коэффициент {raw_ratio:.2f}, язык {language} (фактор {language_factor}), "
            f"скорректированный {adjusted_ratio:.2f}, результат {rate}"
        )
        
        return rate
    
    async def _synthesize_segment_with_semaphore(self, segment, output_path, target_language='kk'):
        """
        Синтезирует речь для одного сегмента с контролем параллелизма
        
        Args:
            segment (SpeechSegment): Сегмент речи
            output_path (str): Путь для сохранения аудиофайла
            target_language (str): Код целевого языка
            
        Returns:
            str: Путь к синтезированному аудиофайлу
        """
        async with self.semaphore:
            start_time = time.time()
            result = await self._synthesize_segment(segment, output_path, target_language)
            elapsed = time.time() - start_time
            
            # Записываем медленные синтезы для анализа
            if elapsed > 5.0:  # более 5 секунд считаем медленным
                self.slow_synthesis_cache.append({
                    'text': segment.translated_text[:100] + ('...' if len(segment.translated_text) > 100 else ''),
                    'time': elapsed,
                    'length': len(segment.translated_text)
                })
                # Оставляем только последние 10 медленных синтезов
                self.slow_synthesis_cache = self.slow_synthesis_cache[-10:]
            
            return result
    
    async def _synthesize_segment(self, segment, output_path, target_language='kk'):
        """
        Синтезирует речь для одного сегмента с высоким качеством и адаптацией
        
        Args:
            segment (SpeechSegment): Сегмент речи
            output_path (str): Путь для сохранения аудиофайла
            target_language (str): Код целевого языка
            
        Returns:
            str: Путь к синтезированному аудиофайлу
        """
        if not segment.translated_text or segment.translated_text.strip() == "":
            logger.warning(f"Отсутствует перевод для сегмента: {segment.text[:30]}...")
            return None
        
        try:
            # Очищаем текст от лишних пробелов
            translated_text = segment.translated_text.strip()
            
            # Подробное логирование о поле говорящего
            logger.info(
                f"Обработка сегмента [{segment.start_time:.2f}s - {segment.end_time:.2f}s], "
                f"определенный пол говорящего: {segment.speaker_gender}, "
                f"уверенность: {getattr(segment, 'gender_confidence', 'не указана')}"
            )
            
            # Выбор голоса с учетом интеллектуального чередования
            voice = self._select_voice(segment.speaker_gender, target_language)
            
            # Длительность оригинального сегмента
            original_duration = segment.end_time - segment.start_time
            
            # Проверяем кэш для этого текста и голоса
            cache_key = f"{translated_text}_{voice}"
            if cache_key in self.tts_cache:
                logger.info(f"Использую кэшированный результат для сегмента: {translated_text[:30]}...")
                cached_data = self.tts_cache[cache_key]
                synthesized_duration = cached_data['duration']
                rate = cached_data['rate']
            else:
                # Создаем временный файл для первой попытки синтеза
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Время начала синтеза
                start_time = time.time()
                
                # Синтезируем речь с нормальной скоростью
                communicate = edge_tts.Communicate(translated_text, voice)
                await communicate.save(temp_path)
                
                # Измеряем время синтеза
                synthesis_time = time.time() - start_time
                logger.debug(f"Время синтеза для сегмента: {synthesis_time:.2f}с")
                
                # Измеряем длительность синтезированной речи
                audio = AudioSegment.from_file(temp_path)
                synthesized_duration = len(audio) / 1000.0  # в секундах
                
                # Рассчитываем оптимальную скорость на основе длительности и характеристик текста
                rate = self._calculate_optimal_speed(
                    original_duration, 
                    synthesized_duration, 
                    translated_text,
                    target_language
                )
                
                # Удаляем временный файл
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {temp_path}: {e}")
                
                # Сохраняем в кэш
                self.tts_cache[cache_key] = {
                    'duration': synthesized_duration,
                    'rate': rate
                }
            
            # Синтезируем речь с оптимальной скоростью
            # Запоминаем выбранные параметры для включения в имя файла
            selected_gender = getattr(self, '_last_selected_gender', 'unknown')
            filename_prefix = f"{selected_gender[0].upper()}"  # M для мужского, F для женского
            
            # Если имя файла не содержит префикс пола, добавляем его
            output_basename = os.path.basename(output_path)
            output_dir = os.path.dirname(output_path)
            
            if not output_basename.startswith(('M_', 'F_', 'U_')):
                new_output_filename = f"{filename_prefix}_{output_basename}"
                output_path = os.path.join(output_dir, new_output_filename)
                
            logger.info(
                f"Синтезируем речь для сегмента "
                f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                f"с голосом {voice} (пол: {selected_gender}) и скоростью {rate}"
            )
            
            # Синтезируем речь с рассчитанной скоростью
            communicate = edge_tts.Communicate(translated_text, voice, rate=rate)
            await communicate.save(output_path)
            
            # Проверяем, что файл создан успешно
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.error(f"Синтезированный файл не был создан или пуст: {output_path}")
                return None
            
            # Проверяем длительность синтезированного аудио для анализа
            try:
                final_audio = AudioSegment.from_file(output_path)
                final_duration = len(final_audio) / 1000.0
                expected_duration = original_duration
                
                # Если сильное отклонение от ожидаемого, логируем для анализа
                if abs(final_duration - expected_duration) > expected_duration * 0.3:  # отклонение >30%
                    logger.warning(
                        f"Значительное отклонение длительности: "
                        f"ожидалось {expected_duration:.2f}с, получено {final_duration:.2f}с "
                        f"для сегмента [{segment.start_time:.2f}s - {segment.end_time:.2f}s]"
                    )
            except Exception as e:
                logger.warning(f"Не удалось проверить длительность синтезированного аудио: {e}")
            
            logger.info(
                f"Синтезирована речь для сегмента "
                f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s], "
                f"исходный пол: {segment.speaker_gender}, выбранный голос: {voice}, "
                f"выбранный пол: {selected_gender}, темп: {rate}"
            )
            
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при синтезе речи для сегмента: {e}")
            return None
    
    async def _process_segments_batch(self, segments, output_dir, target_language):
        """
        Обрабатывает пакет сегментов асинхронно с адаптивным выбором голосов и оптимизацией
        
        Args:
            segments (List[SpeechSegment]): Список сегментов для обработки
            output_dir (str): Директория для сохранения синтезированных аудиофайлов
            target_language (str): Код целевого языка
            
        Returns:
            List[SpeechSegment]: Обновленный список сегментов
        """
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Анализируем сегменты для определения длины и оптимизации последовательности
        total_segments = len(segments)
        
        logger.info(f"Начинаем обработку {total_segments} сегментов для языка {target_language}")
        
        # Сбрасываем счетчики для нового набора сегментов
        self._last_selected_gender = None
        self._consecutive_same_gender_count = 0
        self._segment_counter = 0
        
        # Группируем задачи для оптимизации параллельного выполнения
        # Сначала короткие сегменты, затем более длинные для эффективного использования ресурсов
        segments_with_durations = [(i, s, s.end_time - s.start_time) for i, s in enumerate(segments)]
        
        # Сортируем по длительности (сначала короткие)
        segments_with_durations.sort(key=lambda x: x[2])
        
        # Создаем задачи, все еще сохраняя исходные индексы
        all_tasks = []
        for i, segment, _ in segments_with_durations:
            # Создаем имя файла для синтезированного аудио с учетом пола и порядкового номера
            # Префикс пола (M/F/U) будет добавлен в методе _synthesize_segment
            output_filename = f"synth_{i:04d}_{int(segment.start_time*1000):08d}_{int(segment.end_time*1000):08d}.mp3"
            output_path = os.path.join(output_dir, output_filename)
            
            # Создаем задачу с использованием семафора для ограничения параллелизма
            task = self._synthesize_segment_with_semaphore(segment, output_path, target_language)
            all_tasks.append((i, segment, output_path, task))
            
            logger.info(f"Запланирован синтез сегмента {i+1}/{total_segments}: [{segment.start_time:.2f}s - {segment.end_time:.2f}s]")
        
        # Отображение прогресса и обработка результатов
        completed = 0
        segment_results = [None] * total_segments  # Предварительное заполнение списка результатов
        
        # Используем asyncio.as_completed для обработки результатов по мере их появления
        for i, segment, output_path, task in all_tasks:
            try:
                result = await task
                if result:
                    # Обновляем путь к аудио в сегменте (может отличаться от исходного из-за префикса пола)
                    segment.audio_path = result
                    segment_results[i] = segment  # Сохраняем результат на правильной позиции
                    
                    # Добавляем информацию о выбранном поле как атрибут сегмента
                    selected_gender = getattr(self, '_last_selected_gender', 'unknown')
                    segment.selected_gender = selected_gender
                else:
                    logger.warning(f"Не удалось синтезировать сегмент #{i+1}")
                    segment_results[i] = segment  # Даже если синтез не удался, сохраняем сегмент
                
                # Обновляем прогресс
                completed += 1
                if completed % 5 == 0 or completed == total_segments:  # Каждые 5 сегментов или последний
                    percent_complete = completed / total_segments * 100
                    logger.info(f"Прогресс синтеза: {completed}/{total_segments} сегментов ({percent_complete:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Ошибка при выполнении задачи синтеза для сегмента #{i+1}: {e}")
                segment_results[i] = segment  # Сохраняем сегмент даже при ошибке
        
        # Фильтруем None значения, если они есть
        result_segments = [s for s in segment_results if s is not None]
        
        # Сохраняем кэш для использования в будущих запусках
        self._save_cache()
        
        # Выводим статистику по голосам
        male_count = sum(1 for s in result_segments if hasattr(s, 'selected_gender') and s.selected_gender == 'male')
        female_count = sum(1 for s in result_segments if hasattr(s, 'selected_gender') and s.selected_gender == 'female')
        
        logger.info(f"Завершен синтез всех {len(result_segments)} сегментов: "
                   f"мужские голоса: {male_count}, женские голоса: {female_count}")
        
        # Выводим статистику по медленным синтезам, если они были
        if self.slow_synthesis_cache:
            logger.warning(f"Было {len(self.slow_synthesis_cache)} медленных синтезов. Самый медленный: "
                          f"{max(self.slow_synthesis_cache, key=lambda x: x['time'])['time']:.2f}с "
                          f"для текста длиной {max(self.slow_synthesis_cache, key=lambda x: x['time'])['length']} символов")
        
        return result_segments
    
    async def synthesize_segments(self, segments, output_dir, target_language='kk'):
        """
        Высококачественный синтез речи для всех сегментов с адаптивной синхронизацией
        
        Args:
            segments (List[SpeechSegment]): Список сегментов для синтеза
            output_dir (str): Директория для сохранения синтезированных аудиофайлов
            target_language (str): Код целевого языка
            
        Returns:
            List[SpeechSegment]: Обновленный список сегментов с путями к аудиофайлам
        """
        logger.info(f"Начинаем высококачественный синтез речи для {len(segments)} сегментов на языке {target_language}")
        
        try:
            # Очищаем кэш перед новой партией сегментов, сохраняя только базовые данные
            # Это сохраняет информацию о скорости для повторяющихся фраз, но не занимает много памяти
            pruned_cache = {}
            for key, value in self.tts_cache.items():
                if isinstance(value, dict) and 'duration' in value and 'rate' in value:
                    pruned_cache[key] = {
                        'duration': value['duration'],
                        'rate': value['rate']
                    }
            self.tts_cache = pruned_cache
            
            # Обрабатываем сегменты
            processed_segments = await self._process_segments_batch(segments, output_dir, target_language)
            
            # Проверяем результаты
            success_count = sum(1 for segment in processed_segments if segment.audio_path is not None)
            logger.info(f"Синтез речи завершен для {success_count}/{len(processed_segments)} сегментов")
            
            return processed_segments
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении высококачественного синтеза речи: {e}")
            # В случае ошибки возвращаем исходные сегменты
            return segments 