"""
Модуль для создания итогового видео с синхронизированными аудиосегментами
"""
import os
import logging
import tempfile
import subprocess
from typing import List, Optional
import json
import ffmpeg
from pydub import AudioSegment
import shutil
import uuid
import math

from models.speech_segment import SpeechSegment
from services.tts_manager import TTSManager

logger = logging.getLogger(__name__)

class VideoComposer:
    """
    Класс для создания видео с синхронизированными аудиосегментами
    """
    def __init__(self, codec='libx264', bitrate='8000k'):
        """
        Инициализация композитора видео
        
        Args:
            codec (str): Кодек для выходного видео
            bitrate (str): Битрейт для выходного видео
        """
        self.codec = codec
        self.bitrate = bitrate
        self.tts_manager = TTSManager()
        logger.info("Инициализирован композитор видео")

    def _get_temp_dir(self):
        """
        Создает новую временную директорию в папке проекта
        
        Returns:
            str: Путь к временной директории
        """
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _cleanup_temp_dir(self, temp_dir):
        """
        Очищает временную директорию
        
        Args:
            temp_dir (str): Путь к временной директории
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Не удалось очистить временную директорию {temp_dir}: {e}")

    def _create_composite_audio(
        self,
        segments: List[SpeechSegment],
        background_audio_path: str,
        background_volume: float = 0.1,
        temp_dir: str = None
    ) -> Optional[str]:
        """
        Создание высококачественного композитного аудио из сегментов и фонового аудио
        с интеллектуальным наложением и обработкой
        
        Args:
            segments (List[SpeechSegment]): Список сегментов речи
            background_audio_path (str): Путь к фоновому аудио
            background_volume (float): Громкость фонового аудио (0.0-1.0)
            temp_dir (str): Путь к временной директории
            
        Returns:
            Optional[str]: Путь к созданному композитному аудио
        """
        try:
            # Проверяем наличие сегментов с синтезированной речью
            valid_segments = [s for s in segments if s.audio_path and os.path.exists(s.audio_path)]
            
            if not valid_segments:
                logger.error("Нет сегментов с синтезированной речью для создания композитного аудио")
                return None
            
            logger.info(f"Создание высококачественного композитного аудио из {len(valid_segments)} сегментов")
            
            # Загружаем фоновое аудио
            try:
                background = AudioSegment.from_file(background_audio_path)
                # Нормализуем фоновое аудио
                background = self._normalize_audio(background)
                # Уменьшаем громкость фонового аудио (логарифмическая шкала)
                background_reduction = min(max(int(20 * (1 - background_volume)), 6), 30)
                background = background - background_reduction
                logger.debug(f"Загружено фоновое аудио: {len(background)/1000:.2f} сек, уменьшение громкости: {background_reduction}dB")
            except Exception as e:
                logger.error(f"Ошибка при загрузке фонового аудио: {e}")
                # В случае ошибки создаем пустой фон (тишину)
                background = AudioSegment.silent(duration=int((max([s.end_time for s in valid_segments]) + 2.0) * 1000))
                logger.warning(f"Создан пустой фон длительностью {len(background)/1000:.2f} сек")
            
            # Определяем параметры композиции
            max_end_time = max([s.end_time for s in valid_segments]) + 2.0  # добавляем 2 секунды в конце
            target_duration_ms = int(max_end_time * 1000)
            
            # Предварительно анализируем аудио сегменты
            segment_audio_data = {}
            max_speech_volume = -100  # начальное значение в дБ
            
            # Загружаем и анализируем все сегменты для определения оптимальной громкости
            for segment in valid_segments:
                try:
                    # Создаем уникальный идентификатор сегмента на основе времени
                    segment_key = f"{segment.start_time:.2f}_{segment.end_time:.2f}"
                    
                    # Загружаем аудио сегмента
                    segment_audio = AudioSegment.from_file(segment.audio_path)
                    
                    # Нормализуем громкость сегмента
                    segment_audio = self._normalize_audio(segment_audio)
                    
                    # Сохраняем данные сегмента
                    segment_audio_data[segment_key] = {
                        'audio': segment_audio,
                        'loudness': self._get_audio_loudness(segment_audio),
                        'duration': len(segment_audio) / 1000.0  # в секундах
                    }
                    
                    # Обновляем максимальную громкость
                    if segment_audio_data[segment_key]['loudness'] > max_speech_volume:
                        max_speech_volume = segment_audio_data[segment_key]['loudness']
                    
                except Exception as e:
                    logger.error(f"Ошибка при загрузке аудио сегмента [{segment.start_time:.2f}s - {segment.end_time:.2f}s]: {e}")
            
            # Настраиваем фоновое аудио
            background_duration_ms = len(background)
            
            # Зацикливаем фоновое аудио, если оно короче необходимой длительности
            if background_duration_ms < target_duration_ms:
                # Зацикливание с плавным переходом между копиями
                final_background = self._loop_background_audio(background, target_duration_ms)
            else:
                # Обрезаем до нужной длительности
                final_background = background[:target_duration_ms]
            
            # Создаем финальное композитное аудио на основе финального фона
            composite_audio = final_background
            
            # Создаем карту перекрытия сегментов для определения мест, где нужно дополнительно снизить громкость фона
            overlap_map = self._create_overlap_map(valid_segments, target_duration_ms)
            
            # Заранее определяем места, где нужно снизить громкость фона
            # Создаем "коридоры громкости" для речи
            background_volume_map = self._create_volume_map(valid_segments, overlap_map, target_duration_ms)
            
            # Применяем карту громкости к фону перед наложением речи
            composite_audio = self._apply_volume_map(composite_audio, background_volume_map)
            
            # Добавляем каждый синтезированный сегмент на своем временном участке
            for segment in valid_segments:
                try:
                    # Создаем ключ для поиска данных сегмента
                    segment_key = f"{segment.start_time:.2f}_{segment.end_time:.2f}"
                    
                    # Пропускаем сегменты без аудио данных
                    if segment_key not in segment_audio_data:
                        logger.warning(f"Сегмент [{segment.start_time:.2f}s - {segment.end_time:.2f}s] не найден в кэше данных")
                        continue
                    
                    segment_audio = segment_audio_data[segment_key]['audio']
                    
                    # Рассчитываем позицию сегмента в миллисекундах
                    position_ms = int(segment.start_time * 1000)
                    
                    # Проверяем, что позиция не выходит за границы композитного аудио
                    if position_ms >= len(composite_audio):
                        logger.warning(
                            f"Сегмент [{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                            f"выходит за границы композитного аудио ({len(composite_audio)/1000:.2f}s)"
                        )
                        continue
                    
                    # Добавляем плавное нарастание и затухание для избежания щелчков
                    segment_audio = segment_audio.fade_in(30).fade_out(30)
                    
                    # Добавляем небольшую задержку для более естественного звучания, если сегмент короткий
                    segment_duration = len(segment_audio)
                    if segment_duration < 500:  # если сегмент короче 0.5 секунды
                        # Добавляем небольшую паузу в начале для избежания резкого начала
                        segment_audio = AudioSegment.silent(duration=20) + segment_audio
                    
                    # Накладываем сегмент в нужную позицию
                    composite_audio = composite_audio.overlay(segment_audio, position=position_ms)
                    
                    logger.info(
                        f"Добавлен сегмент [{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                        f"длительностью {segment_duration/1000:.2f}s в композитное аудио"
                    )
                    
                except Exception as e:
                    logger.error(f"Ошибка при добавлении сегмента в композитное аудио: {e}")
                    continue
            
            # Финальная обработка звука
            # Применяем компрессию для выравнивания громкости
            composite_audio = self._apply_dynamic_compression(composite_audio)
            
            # Путь для сохранения композитного аудио
            output_path = os.path.join(temp_dir, "composite_audio.wav")
            
            # Экспортируем композитное аудио в файл с высоким качеством
            composite_audio.export(
                output_path, 
                format="wav",
                parameters=["-ar", "48000", "-ac", "2", "-b:a", "320k"]
            )
            
            logger.info(f"Создано высококачественное композитное аудио: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Критическая ошибка при создании композитного аудио: {e}")
            return None
    
    def _normalize_audio(self, audio: AudioSegment, target_dBFS=-18.0, max_gain_db=10.0) -> AudioSegment:
        """
        Нормализует аудио до целевой громкости с ограничением максимального усиления
        
        Args:
            audio (AudioSegment): Аудио для нормализации
            target_dBFS (float): Целевая громкость в dBFS
            max_gain_db (float): Максимальное усиление в dB
            
        Returns:
            AudioSegment: Нормализованное аудио
        """
        if len(audio) == 0:
            return audio
            
        # Текущая громкость
        current_dBFS = audio.dBFS
        
        # Если текущая громкость слишком низкая, не усиливаем больше максимального усиления
        if current_dBFS < -80:  # Очень тихое или беззвучное аудио
            return audio
            
        # Вычисляем необходимое изменение громкости
        dBFS_change = target_dBFS - current_dBFS
        
        # Ограничиваем максимальное усиление
        if dBFS_change > max_gain_db:
            dBFS_change = max_gain_db
            
        # Применяем изменение громкости
        return audio.apply_gain(dBFS_change)
    
    def _get_audio_loudness(self, audio: AudioSegment) -> float:
        """
        Получает средний уровень громкости аудио
        
        Args:
            audio (AudioSegment): Аудио для анализа
            
        Returns:
            float: Уровень громкости в dBFS
        """
        if len(audio) == 0:
            return -100.0
            
        return audio.dBFS
    
    def _loop_background_audio(self, background: AudioSegment, target_duration_ms: int) -> AudioSegment:
        """
        Зацикливает фоновое аудио с плавными переходами
        
        Args:
            background (AudioSegment): Фоновое аудио
            target_duration_ms (int): Требуемая длительность в миллисекундах
            
        Returns:
            AudioSegment: Зацикленное аудио требуемой длительности
        """
        bg_duration = len(background)
        
        # Если фон слишком короткий (менее 5 секунд), дублируем его несколько раз перед зацикливанием
        if bg_duration < 5000:
            # Дублируем фон 3 раза
            background = background * 3
            bg_duration = len(background)
        
        # Если фон по-прежнему короче требуемой длительности, зацикливаем его
        if bg_duration < target_duration_ms:
            # Определяем количество необходимых повторений (с запасом)
            repetitions = (target_duration_ms // bg_duration) + 1
            
            # Создаем плавный переход между копиями (кроссфейд)
            crossfade_ms = min(1000, bg_duration // 4)  # Переход не более 1 секунды или четверти длины фона
            
            # Создаем итоговый фон с кроссфейдами между копиями
            result = background
            for _ in range(repetitions - 1):
                result = result.append(background, crossfade=crossfade_ms)
            
            # Обрезаем до требуемой длительности
            result = result[:target_duration_ms]
            return result
        else:
            # Если фон длиннее, просто обрезаем его
            return background[:target_duration_ms]
    
    def _create_overlap_map(self, segments: List[SpeechSegment], duration_ms: int) -> List[int]:
        """
        Создает карту перекрытия сегментов для каждой миллисекунды
        
        Args:
            segments (List[SpeechSegment]): Список сегментов
            duration_ms (int): Длительность в миллисекундах
            
        Returns:
            List[int]: Карта перекрытия (количество сегментов в каждой миллисекунде)
        """
        # Инициализируем карту нулями
        overlap_map = [0] * duration_ms
        
        # Для каждого сегмента увеличиваем счетчики в соответствующем диапазоне
        for segment in segments:
            start_ms = int(segment.start_time * 1000)
            end_ms = int(segment.end_time * 1000)
            
            # Проверяем границы
            if start_ms < 0:
                start_ms = 0
            if end_ms > duration_ms:
                end_ms = duration_ms
            if start_ms >= end_ms:
                continue
                
            # Увеличиваем счетчики в диапазоне сегмента
            for i in range(start_ms, min(end_ms, duration_ms)):
                overlap_map[i] += 1
                
        return overlap_map
    
    def _create_volume_map(self, segments: List[SpeechSegment], overlap_map: List[int], duration_ms: int) -> List[float]:
        """
        Создает карту громкости фона для каждой миллисекунды
        
        Args:
            segments (List[SpeechSegment]): Список сегментов
            overlap_map (List[int]): Карта перекрытия сегментов
            duration_ms (int): Длительность в миллисекундах
            
        Returns:
            List[float]: Карта громкости (коэффициент громкости для каждой миллисекунды)
        """
        # Инициализируем карту единицами (без изменения громкости)
        volume_map = [1.0] * duration_ms
        
        # Для каждого сегмента создаем "коридор" с пониженной громкостью
        for segment in segments:
            start_ms = max(0, int((segment.start_time - 0.2) * 1000))  # 200ms перед началом
            end_ms = min(duration_ms, int((segment.end_time + 0.2) * 1000))  # 200ms после конца
            
            if start_ms >= end_ms or start_ms >= duration_ms or end_ms <= 0:
                continue
                
            # Коэффициент снижения громкости зависит от перекрытия с другими сегментами
            for i in range(start_ms, end_ms):
                if i < 0 or i >= duration_ms:
                    continue
                    
                # Базовое снижение громкости для речи
                reduction = 0.25  # до 25% от исходной громкости (или -12dB)
                
                # Если есть перекрытие с другими сегментами, снижаем сильнее
                if overlap_map[i] > 1:
                    reduction = 0.15  # до 15% от исходной громкости (или -16dB)
                
                # Создаем плавный переход в начале и конце сегмента
                fade_duration_ms = 200  # 200ms для fade in/out
                
                if i < start_ms + fade_duration_ms:  # fade in
                    fade_factor = (i - start_ms) / fade_duration_ms
                    current_reduction = 1.0 - ((1.0 - reduction) * fade_factor)
                elif i > end_ms - fade_duration_ms:  # fade out
                    fade_factor = (end_ms - i) / fade_duration_ms
                    current_reduction = 1.0 - ((1.0 - reduction) * fade_factor)
                else:  # основная часть
                    current_reduction = reduction
                
                # Обновляем карту громкости с минимальным значением (наиболее сильное снижение)
                volume_map[i] = min(volume_map[i], current_reduction)
                
        return volume_map
    
    def _apply_volume_map(self, audio: AudioSegment, volume_map: List[float]) -> AudioSegment:
        """
        Применяет карту громкости к аудио
        
        Args:
            audio (AudioSegment): Исходное аудио
            volume_map (List[float]): Карта громкости
            
        Returns:
            AudioSegment: Обработанное аудио
        """
        # Если карта громкости или аудио пустые, возвращаем исходное аудио
        if not volume_map or len(audio) == 0:
            return audio
            
        # Проверяем размеры
        audio_duration_ms = len(audio)
        if len(volume_map) < audio_duration_ms:
            # Расширяем карту до длины аудио
            volume_map = volume_map + [1.0] * (audio_duration_ms - len(volume_map))
        elif len(volume_map) > audio_duration_ms:
            # Обрезаем карту до длины аудио
            volume_map = volume_map[:audio_duration_ms]
        
        # Размер сегмента для обработки (в миллисекундах)
        # Выбираем значение, при котором будет не слишком много сегментов, но и детализация не пострадает
        segment_size_ms = 50  # 50ms
        
        # Разбиваем аудио на сегменты и применяем соответствующую громкость
        result = AudioSegment.empty()
        for start_ms in range(0, audio_duration_ms, segment_size_ms):
            end_ms = min(start_ms + segment_size_ms, audio_duration_ms)
            
            # Извлекаем сегмент аудио
            segment = audio[start_ms:end_ms]
            
            # Вычисляем среднюю громкость для данного сегмента
            segment_volume = sum(volume_map[start_ms:end_ms]) / (end_ms - start_ms)
            
            # Применяем изменение громкости
            if segment_volume < 1.0:
                # Преобразуем коэффициент громкости в дБ (логарифмическая шкала)
                volume_db = 20 * math.log10(segment_volume)
                segment = segment.apply_gain(volume_db)
            
            # Добавляем обработанный сегмент к результату
            result += segment
        
        return result
    
    def _apply_dynamic_compression(self, audio: AudioSegment, threshold=-18.0, ratio=2.5, attack_ms=5, release_ms=100) -> AudioSegment:
        """
        Применяет динамическую компрессию к аудио для выравнивания громкости
        
        Args:
            audio (AudioSegment): Исходное аудио
            threshold (float): Порог срабатывания компрессора в dBFS
            ratio (float): Степень компрессии
            attack_ms (int): Время атаки в миллисекундах
            release_ms (int): Время восстановления в миллисекундах
            
        Returns:
            AudioSegment: Обработанное аудио
        """
        # Реализация упрощенной компрессии на уровне Python
        # Для продакшена рекомендуется использовать специализированные аудио библиотеки
        
        # В данной реализации просто немного выравниваем общую громкость
        # и добавляем небольшое усиление
        
        # Нормализуем к целевой громкости
        normalized = self._normalize_audio(audio, target_dBFS=threshold-3, max_gain_db=6.0)
        
        # Добавляем мягкую лимитацию для предотвращения искажений
        # (просто понижаем громкость на 1dB для безопасности)
        limited = normalized.apply_gain(-1.0)
        
        return limited

    def _create_srt_subtitles(
        self,
        segments: List[SpeechSegment],
        output_path: str
    ) -> Optional[str]:
        """
        Создание файла субтитров в формате SRT
        
        Args:
            segments (List[SpeechSegment]): Список сегментов речи
            output_path (str): Путь для сохранения файла субтитров
            
        Returns:
            Optional[str]: Путь к созданному файлу субтитров
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    # Форматируем время в формат SRT (HH:MM:SS,mmm)
                    start_time = self._format_time(segment.start_time)
                    end_time = self._format_time(segment.end_time)
                    
                    # Записываем субтитр
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.translated_text}\n\n")
            
            logger.info(f"Создан файл субтитров: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при создании файла субтитров: {e}")
            return None

    def _format_time(self, seconds: float) -> str:
        """
        Форматирование времени в формат SRT
        
        Args:
            seconds (float): Время в секундах
            
        Returns:
            str: Отформатированное время в формате HH:MM:SS,mmm
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"

    def _add_subtitles_to_video(self, video_path: str, subtitles_path: str, output_path: str, project_paths) -> str:
        """
        Добавляет субтитры к видео используя ffmpeg
        
        Args:
            video_path (str): Путь к видео
            subtitles_path (str): Путь к файлу субтитров
            output_path (str): Путь для сохранения результата
            project_paths: Объект ProjectPaths для управления путями
            
        Returns:
            str: Путь к видео с субтитрами
        """
        try:
            # Создаем уникальные имена для временных файлов
            temp_id = str(uuid.uuid4())[:8]
            temp_video = os.path.join(project_paths.output_dir, f"temp_video_{temp_id}.mp4")
            temp_subs = os.path.join(project_paths.subtitles_dir, f"temp_subs_{temp_id}.srt")
            temp_output = os.path.join(project_paths.output_dir, f"temp_output_{temp_id}.mp4")
            
            # Создаем директории если их нет
            os.makedirs(os.path.dirname(temp_video), exist_ok=True)
            os.makedirs(os.path.dirname(temp_subs), exist_ok=True)
            
            # Копируем файлы с проверкой на существование
            if os.path.exists(video_path):
                shutil.copy2(video_path, temp_video)
            else:
                logger.error(f"Исходное видео не найдено: {video_path}")
                return video_path
                
            if os.path.exists(subtitles_path):
                shutil.copy2(subtitles_path, temp_subs)
            else:
                logger.error(f"Файл субтитров не найден: {subtitles_path}")
                return video_path
            
            # Формируем команду ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-vf", f"subtitles={temp_subs}",
                "-c:v", self.codec,
                "-b:v", self.bitrate,
                "-c:a", "copy",
                temp_output
            ]
            
            # Выполняем команду
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Ошибка ffmpeg при добавлении субтитров: {result.stderr}")
                return video_path
            
            # Проверяем результат и копируем в целевой путь
            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                shutil.copy2(temp_output, output_path)
                logger.info(f"Субтитры успешно добавлены в видео: {output_path}")
                return output_path
            else:
                logger.error("Не удалось создать видео с субтитрами")
                return video_path
                
        except Exception as e:
            logger.error(f"Ошибка при добавлении субтитров: {e}")
            return video_path
        finally:
            # Очищаем временные файлы
            for temp_file in [temp_video, temp_subs, temp_output]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {temp_file}: {e}")

    def compose_video(
        self,
        video_path: str,
        segments: List[SpeechSegment],
        output_path: str,
        include_subtitles: bool = True,
        create_subtitles_file: bool = True,
        project_paths = None
    ) -> Optional[str]:
        """
        Создает итоговое видео с синхронизированными аудиосегментами и отдельным файлом субтитров
        
        Args:
            video_path (str): Путь к исходному видео
            segments (List[SpeechSegment]): Список сегментов речи
            output_path (str): Путь или имя файла для сохранения итогового видео
            include_subtitles (bool): Создавать ли файл субтитров
            create_subtitles_file (bool): Создавать ли отдельный файл субтитров (оставлено для обратной совместимости)
            project_paths: Объект ProjectPaths для управления путями
            
        Returns:
            Optional[str]: Путь к итоговому видео
        """
        logger.info("Начинаем создание итогового видео")
        
        try:
            # Проверяем корректность путей
            if not video_path or not os.path.exists(video_path):
                raise ValueError(f"Исходное видео не найдено или путь некорректен: {video_path}")
                
            if not output_path:
                raise ValueError("Не указан путь для сохранения итогового видео")
                
            if not project_paths:
                raise ValueError("Не предоставлен объект ProjectPaths")
            
            # Если передано только имя файла без пути, сохраняем в output_dir проекта
            if not os.path.dirname(output_path):
                output_path = os.path.join(project_paths.output_dir, output_path)
                logger.info(f"Используем путь для сохранения: {output_path}")
            
            # Создаем директорию для выходного файла
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Создана директория для выходного файла: {output_dir}")
            
            # Шаг 1: Создаем композитную аудиодорожку
            composite_audio = self._create_composite_audio(
                segments,
                self._extract_audio(video_path, project_paths.paths['background_audio']),
                temp_dir=project_paths.audio_dir
            )
            
            if not composite_audio:
                raise RuntimeError("Не удалось создать композитное аудио")
            
            # Добавляем аудио к видео
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", composite_audio,
                "-c:v", "copy",
                "-c:a", "aac",
                "-strict", "experimental",
                "-map", "0:v:0",
                "-map", "1:a:0",
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Ошибка при добавлении аудио: {result.stderr}")
                raise RuntimeError("Не удалось создать видео с дубляжом")
            
            # Создаем файл субтитров, если требуется
            if include_subtitles:
                srt_output = output_path.rsplit('.', 1)[0] + '.srt'
                if self._create_srt_subtitles(segments, srt_output):
                    logger.info(f"Создан файл субтитров: {srt_output}")
                else:
                    logger.error("Не удалось создать файл субтитров")
            
            # Проверяем результат
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Создание итогового видео завершено успешно: {output_path}")
                return output_path
            else:
                raise RuntimeError("Не удалось создать итоговое видео")
                
        except Exception as e:
            logger.error(f"Ошибка при создании итогового видео: {e}")
            raise

    def _extract_audio(self, video_path: str, output_path: str) -> Optional[str]:
        """
        Извлекает аудиодорожку из видео с помощью ffmpeg
        
        Args:
            video_path (str): Путь к видеофайлу
            output_path (str): Путь для сохранения аудиодорожки
            
        Returns:
            Optional[str]: Путь к извлеченной аудиодорожке
        """
        try:
            # Создаем директорию для аудио если её нет
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Извлекаем аудио из видео
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=2, ar='44100')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f"Аудиодорожка успешно извлечена: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при извлечении аудиодорожки: {e}")
            return None
    
    def _get_video_info(self, video_path):
        """
        Получает информацию о видеофайле с помощью ffmpeg
        
        Args:
            video_path (str): Путь к видеофайлу
            
        Returns:
            dict: Информация о видео (длительность, fps, etc.)
        """
        try:
            # Используем ffprobe для получения информации о видео
            probe = ffmpeg.probe(video_path)
            
            # Извлекаем информацию о видеопотоке
            video_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                logger.error(f"Не найден видеопоток в файле: {video_path}")
                return {"duration": 0, "width": 0, "height": 0, "fps": 0}
            
            # Вычисляем FPS
            fps_parts = video_stream.get('r_frame_rate', '25/1').split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1] if len(fps_parts) > 1 else 1)
            
            # Извлекаем остальную информацию
            info = {
                "duration": float(probe.get('format', {}).get('duration', 0)),
                "width": int(video_stream.get('width', 0)),
                "height": int(video_stream.get('height', 0)),
                "fps": fps
            }
            
            logger.info(f"Получена информация о видео: {info}")
            return info
        except Exception as e:
            logger.error(f"Ошибка при получении информации о видео: {e}")
            return {"duration": 0, "width": 0, "height": 0, "fps": 25} 