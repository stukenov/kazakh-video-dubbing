"""
Основной модуль для системы дубляжа видео с английского на казахский
с синхронизацией субтитров, перевода и аудио
"""
import os
# Отключаем предупреждения TensorFlow о oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import logging
import argparse
import asyncio
from pathlib import Path
import uuid
from typing import Dict, Any, Optional, List

# Корректный импорт для OpenAI Whisper
try:
    import whisper
except ImportError:
    logging.error("Не удалось импортировать whisper. Установите пакет openai-whisper.")
    whisper = None
    sys.exit(1)

# Используем ffmpeg-python
import ffmpeg
import torch

# Импортируем компоненты системы
from core.translator import Translator
from core.transcriber import WhisperTranscriber
from core.speech_synthesizer import EdgeTTSSynthesizer
from core.video_processor import VideoProcessor
from core.speech_segmenter import SpeechSegmenter
from models.speech_segment import SpeechSegment
from core.synchronized_synthesizer import SynchronizedSynthesizer
from core.video_composer import VideoComposer
from services.tts_manager import TTSManager
from core.subtitle_synchronizer import SubtitleSynchronizer
from core.audio_processor import AudioProcessor
from services.tts_services import YandexTTSService, EdgeTTSService
from services.whisper_service import WhisperService

# Импортируем настройки
from utils.config import Config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dubbing_full_log.log", mode='w', encoding='utf-8')
    ]
)

# Включаем подробное логирование для модуля subtitle_synchronizer
logging.getLogger('modules.subtitle_synchronizer').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

class ProjectPaths:
    """
    Класс для управления путями проекта
    """
    def __init__(self, project_id: str = None):
        """
        Инициализация путей проекта
        
        Args:
            project_id (str, optional): ID проекта. Если не указан, генерируется автоматически
        """
        if project_id is None:
            project_id = str(uuid.uuid4())
        
        # Корневая директория проекта
        self.project_root = os.getcwd()
        
        # Основные директории
        self.temp_root = os.path.join(self.project_root, "temp_files")
        self.project_dir = os.path.join(self.temp_root, project_id)
        
        # Поддиректории для разных этапов обработки
        self.audio_dir = os.path.join(self.project_dir, "audio")
        self.segments_dir = os.path.join(self.project_dir, "segments")
        self.synthesis_dir = os.path.join(self.project_dir, "synthesis")
        self.subtitles_dir = os.path.join(self.project_dir, "subtitles")
        self.output_dir = os.path.join(self.project_dir, "output")
        
        # Создаем все директории
        for directory in [self.temp_root, self.project_dir, self.audio_dir, 
                         self.segments_dir, self.synthesis_dir, self.subtitles_dir,
                         self.output_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Пути к файлам для каждого этапа
        self.paths = {
            # Аудио файлы
            'extracted_audio': os.path.join(self.audio_dir, "extracted_audio.wav"),
            'composite_audio': os.path.join(self.audio_dir, "composite_audio.wav"),
            'background_audio': os.path.join(self.audio_dir, "background_audio.wav"),
            
            # Сегменты
            'segments_json': os.path.join(self.segments_dir, "segments.json"),
            'segments_audio': os.path.join(self.segments_dir, "segments"),
            
            # Синтезированная речь
            'synthesis_output': os.path.join(self.synthesis_dir, "synthesized"),
            
            # Субтитры
            'subtitles_srt': os.path.join(self.subtitles_dir, "subtitles.srt"),
            'subtitles_ass': os.path.join(self.subtitles_dir, "subtitles.ass"),
            
            # Промежуточные видео файлы
            'temp_video': os.path.join(self.output_dir, "temp_video.mp4"),
            'video_with_audio': os.path.join(self.output_dir, "video_with_audio.mp4"),
            'final_video': os.path.join(self.output_dir, "final_video.mp4")
        }
        
        logger.info(f"Инициализированы пути проекта {project_id}")
        logger.info(f"Корневая директория проекта: {self.project_root}")
        logger.info(f"Директория временных файлов: {self.project_dir}")

class SynchronizedVideoDubber:
    """
    Основной класс для дубляжа видео с синхронизацией
    """
    def __init__(self, translator: Translator = None, config: Dict[str, Any] = None):
        """Инициализация системы дубляжа"""
        self.config = Config() if config is None else config
        
        # Создаем глобальный цикл событий для асинхронных операций
        self.loop = asyncio.get_event_loop()
        
        # Инициализация компонентов системы
        self.segmenter = SpeechSegmenter(
            model_name=self.config.SPEECH_RECOGNITION['model'],
            language=self.config.SPEECH_RECOGNITION['language']
        )
        
        # Используем переданный translator или создаем новый
        self.translator = translator or Translator(
            source_lang=self.config.TRANSLATION['source_language'],
            target_lang=self.config.TRANSLATION['target_language']
        )
        
        # Создаем менеджер TTS
        self.tts_manager = TTSManager(self.config)
        
        # Создаем синтезатор с менеджером TTS
        self.synthesizer = SynchronizedSynthesizer(tts_manager=self.tts_manager)
        
        # Для обратной совместимости сохраняем базовые компоненты
        self.video_processor = VideoProcessor(
            audio_sample_rate=self.config.AUDIO_PROCESSING['sample_rate'],
            codec=self.config.VIDEO_PROCESSING['codec'],
            bitrate=self.config.VIDEO_PROCESSING['bitrate']
        )
        
        # Создаем композитор видео и передаем ему tts_manager
        self.video_composer = VideoComposer(
            codec=self.config.VIDEO_PROCESSING['codec'],
            bitrate=self.config.VIDEO_PROCESSING['bitrate']
        )
        # Заменяем tts_manager в video_composer на наш экземпляр
        self.video_composer.tts_manager = self.tts_manager
        
        # Инициализируем синхронизатор субтитров с API ключом
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.subtitle_synchronizer = SubtitleSynchronizer(
            min_subtitle_duration=self.config.SUBTITLES.get('min_duration', 1.0),
            max_subtitle_duration=self.config.SUBTITLES.get('max_duration', 7.0),
            min_gap=self.config.SUBTITLES.get('min_gap', 0.1),
            target_char_rate=self.config.SUBTITLES.get('char_rate'),
            openai_api_key=openai_api_key
        )
        
        logger.info("Система синхронизированного дубляжа инициализирована")
    
    async def process_video(self, input_path, output_path, sync_hq=True):
        """
        Обработка видео с синхронизацией или без
        
        Args:
            input_path (str): Путь к входному видеофайлу
            output_path (str): Путь для сохранения обработанного видео
            sync_hq (bool): Использовать ли режим высокого качества синхронизации
            
        Returns:
            str: Путь к обработанному видео
        """
        return await self._process_video_synchronized(input_path, output_path, sync_hq)
    
    async def _process_video_simple(self, input_path, output_path):
        """
        Простая обработка видео без синхронизации (для обратной совместимости)
        
        Args:
            input_path (str): Путь к входному видеофайлу
            output_path (str): Путь для сохранения обработанного видео
            
        Returns:
            str: Путь к обработанному видео
        """
        try:
            # Создаем объект для управления путями проекта
            project_paths = ProjectPaths()
            
            # Шаг 1: Извлечение аудио из видео
            audio_path = project_paths.paths['extracted_audio']
            self.video_processor.extract_audio(input_path, audio_path)
            
            # Шаг 2: Транскрипция английской речи в текст
            transcript = self.segmenter.model.transcribe(audio_path, language='en')["text"]
            logger.info(f"Распознанный текст: {transcript[:100]}...")
            
            # Шаг 3: Перевод текста с английского на казахский
            translated_text = await self.translator.translate(transcript)
            logger.info(f"Переведенный текст: {translated_text[:100]}...")
            
            # Шаг 4: Синтез речи на казахском
            dubbed_audio_path = os.path.join(project_paths.synthesis_dir, "full_audio.wav")
            os.makedirs(os.path.dirname(dubbed_audio_path), exist_ok=True)
            
            # Создаем один сегмент на все аудио
            segment = SpeechSegment(
                text=transcript,
                start_time=0,
                end_time=999999,  # Большое значение для охвата всего видео
                speaker_gender='unknown',
                translated_text=translated_text
            )
            
            # Синтезируем речь
            segments = await self.synthesizer.synthesize_segments(
                [segment],
                project_paths.paths['synthesis_output'],
                self.config.TRANSLATION['target_language']
            )
            
            # Проверяем, что синтез прошел успешно и путь к аудио обновлен
            if segments and segments[0].audio_path and os.path.exists(segments[0].audio_path):
                dubbed_audio_path = segments[0].audio_path
                logger.info(f"Синтезирована речь: {dubbed_audio_path}")
            else:
                logger.error("Не удалось синтезировать речь")
                raise RuntimeError("Ошибка синтеза речи")
            
            # Шаг 5: Наложение синтезированной речи на видео
            output_path = self.video_processor.merge_audio_with_video(
                input_path, 
                dubbed_audio_path, 
                output_path,
                background_volume=0.1,
                original_fps=self.config.VIDEO_PROCESSING['fps']
            )
            
            logger.info(f"Простая обработка видео завершена. Результат сохранен: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при обработке видео: {e}")
            raise
    
    async def _process_video_synchronized(self, input_path, output_path, sync_hq=True):
        """
        Обработка видео с синхронизацией речи
        
        Args:
            input_path (str): Путь к входному видеофайлу
            output_path (str): Путь для сохранения обработанного видео
            sync_hq (bool): Использовать ли режим высокого качества синхронизации
            
        Returns:
            str: Путь к обработанному видео
        """
        try:
            # Создаем объект для управления путями проекта
            project_paths = ProjectPaths()
            
            # Настраиваем логирование переводов в директории проекта
            if hasattr(self.translator, 'set_log_dir'):
                self.translator.set_log_dir(project_paths.output_dir)
                logger.info(f"Настроено логирование переводов в директории: {project_paths.output_dir}")
            
            # Шаг 1: Извлечение аудио из видео
            audio_path = project_paths.paths['extracted_audio']
            self.video_processor.extract_audio(input_path, audio_path)
            
            # Шаг 2: Сегментация аудио с определением пола говорящего
            segments = self.segmenter.segment_audio(
                audio_path,
                project_paths.paths['segments_audio']
            )
            
            logger.info(f"Получено {len(segments)} сегментов речи")
            
            # Шаг 3: Перевод текста для каждого сегмента
            for i, segment in enumerate(segments):
                segment.translated_text = await self.translator.translate(segment.text)
                logger.info(
                    f"Переведен сегмент {i+1}/{len(segments)}: "
                    f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                    f"'{segment.text[:30]}{'...' if len(segment.text) > 30 else ''}' -> "
                    f"'{segment.translated_text[:30]}{'...' if len(segment.translated_text) > 30 else ''}'"
                )
            
            # Шаг 4: Смысловая синхронизация субтитров с использованием ChatGPT
            if sync_hq:
                logger.info("Применяем смысловую синхронизацию с использованием ChatGPT")
                segments = await self.subtitle_synchronizer.synchronize_segments_with_gpt(
                    segments,
                    target_language=self.config.TRANSLATION['target_language'],
                    high_quality=True
                )
            else:
                # Традиционная синхронизация без использования GPT
                segments = self.subtitle_synchronizer.synchronize_segments(
                    segments,
                    high_quality=False
                )
            
            # Шаг 5: Синтез речи для каждого сегмента с учетом длительности
            segments = await self.synthesizer.synthesize_segments(
                segments,
                project_paths.paths['synthesis_output'],
                self.config.TRANSLATION['target_language']
            )
            
            # Шаг 6: Дополнительная корректировка на основе синтезированного аудио
            if sync_hq:
                segments = self.subtitle_synchronizer.apply_audio_based_adjustments(
                    segments,
                    project_paths.paths['synthesis_output']
                )
            
            # Шаг 7: Создание композитного аудио
            composite_audio_path = self.video_composer._create_composite_audio(
                segments,
                audio_path,
                background_volume=self.config.VIDEO_PROCESSING['background_volume'],
                temp_dir=project_paths.audio_dir
            )
            
            if not composite_audio_path:
                raise RuntimeError("Не удалось создать композитное аудио")
            
            # Шаг 8: Создание субтитров
            subtitles_path = self.video_composer._create_srt_subtitles(
                segments,
                project_paths.paths['subtitles_srt']
            )
            
            if not subtitles_path:
                logger.warning("Не удалось создать файл субтитров")
            
            # Шаг 9: Наложение аудио и субтитров на видео
            output_path = self.video_composer.compose_video(
                input_path=input_path,
                segments=segments,
                output_path=output_path
            )
            
            logger.info(f"Синхронизированная обработка видео завершена. Результат сохранен: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при обработке видео: {e}")
            raise

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Система дубляжа видео с английского на казахский')
    parser.add_argument('--input', type=str, required=True, help='Путь к входному видео')
    parser.add_argument('--output', type=str, required=True, help='Путь для сохранения результата')
    parser.add_argument('--tts', type=str, choices=['yandex', 'edge'], default='yandex', help='Сервис TTS (yandex или edge)')
    parser.add_argument('--full_llm', type=bool, default=False, help='Использовать полный контекст для перевода через LLM')
    return parser.parse_args()

async def process_video(input_path: str, output_path: str, tts_service: str = 'yandex', full_llm: bool = False):
    """
    Обрабатывает видео: извлекает аудио, транскрибирует речь, переводит и создает дубляж
    
    Args:
        input_path (str): Путь к входному видео
        output_path (str): Путь для сохранения результата
        tts_service (str): Сервис TTS для синтеза речи
        full_llm (bool): Использовать полный контекст для перевода через LLM
    """
    try:
        # Проверяем наличие входного файла
        if not os.path.exists(input_path):
            logger.error(f"Входной файл не найден: {input_path}")
            return False
            
        # Создаем директорию для выходного файла если путь содержит директории
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Начинаем обработку видео: {input_path}")
        logger.info(f"Режим перевода: {'Полный контекст LLM' if full_llm else 'Построчный'}")
        
        # Инициализируем компоненты
        config = Config()
        video_processor = VideoProcessor()
        audio_processor = AudioProcessor()
        translator = Translator()
        
        # Выбираем сервис TTS
        if tts_service == 'yandex':
            tts_manager = YandexTTSService(api_key=config.yandex_speechkit_api_key)
            logger.info("Инициализирован Yandex TTS сервис")
        else:
            tts_manager = EdgeTTSService()
            logger.info("Инициализирован Edge TTS сервис")
            
        synthesizer = SynchronizedSynthesizer(tts_manager=tts_manager)
        
        # Извлекаем аудио из видео
        logger.info("Извлекаем аудио из видео...")
        audio_path = os.path.join("temp_files", "extracted_audio.wav")
        video_processor.extract_audio(input_path, audio_path)
        
        # Транскрибируем аудио
        logger.info("Транскрибируем аудио...")
        whisper = WhisperService()
        result = whisper.transcribe(audio_path)
        segments = result.get("segments", [])
        
        if full_llm:
            # Собираем весь текст из сегментов
            full_text = "\n".join([segment["text"] for segment in segments])
            logger.info(f"Собран полный текст для перевода ({len(full_text)} символов)")
            
            # Переводим весь текст целиком
            translated_full_text = await translator.translate_full_context(
                full_text,
                source_lang="en",
                target_lang="kk"
            )
            logger.info("Получен полный перевод текста")
            
            # Разбиваем переведенный текст на сегменты с сохранением таймингов
            translated_segments = await translator.align_translation_with_timings(
                original_segments=segments,
                translated_full_text=translated_full_text
            )
            logger.info("Перевод разбит на сегменты с сохранением таймингов")
        else:
            # Переводим каждый сегмент отдельно
            translated_segments = []
            for segment in segments:
                translated_text = await translator.translate(
                    segment["text"],
                    source_lang="en",
                    target_lang="kk"
                )
                translated_segments.append(SpeechSegment(
                    text=translated_text,
                    start_time=segment["start"],
                    end_time=segment["end"],
                    speaker_gender='unknown',
                    translated_text=None,
                    audio_path=None
                ))
        
        # Создаем синтезатор речи
        logger.info("Синтезируем речь для переведенных сегментов...")
        audio_segments = await synthesizer.synthesize_segments(
            translated_segments,
            output_dir=os.path.join("temp_files", "synthesized_audio"),
            language='kk'
        )
        
        # Собираем финальное видео
        logger.info("Собираем финальное видео...")
        video_composer = VideoComposer()
        await video_composer.compose_video(
            input_path=input_path,
            segments=audio_segments,
            output_path=output_path
        )
        
        logger.info(f"Видео успешно обработано и сохранено: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {str(e)}", exc_info=True)
        return False

async def main():
    """Основная функция"""
    args = parse_args()
    await process_video(args.input, args.output, args.tts, args.full_llm)

if __name__ == "__main__":
    asyncio.run(main()) 