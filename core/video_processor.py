"""
Модуль для обработки видео - извлечение аудио и объединение с синтезированной речью
"""
import os
import logging
from typing import Optional, Dict, Any

import ffmpeg

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Класс для обработки видео - извлечение аудио и наложение синтезированной речи
    """
    def __init__(self, audio_sample_rate: int = 24000, codec: str = 'libx264', bitrate: str = '8000k'):
        """
        Инициализация обработчика видео
        
        Args:
            audio_sample_rate (int): Частота дискретизации аудио
            codec (str): Видеокодек
            bitrate (str): Битрейт видео
        """
        self.audio_sample_rate = audio_sample_rate
        self.codec = codec
        self.bitrate = bitrate
        logger.info("Инициализирован обработчик видео")
    
    def extract_audio(self, video_path: str, output_path: str) -> Optional[str]:
        """
        Извлечение аудио из видео
        
        Args:
            video_path (str): Путь к видео
            output_path (str): Путь для сохранения аудио
            
        Returns:
            Optional[str]: Путь к извлеченному аудио
        """
        try:
            # Создаем директорию для аудио если её нет
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Извлекаем аудио
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_path,
                    acodec='pcm_s16le',
                    ac=1,
                    ar=str(self.audio_sample_rate)
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f"Аудио успешно извлечено: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Ошибка при извлечении аудио: {e.stderr.decode()}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при извлечении аудио: {e}")
            return None
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о видео
        
        Args:
            video_path (str): Путь к видео
            
        Returns:
            Optional[Dict[str, Any]]: Информация о видео
        """
        try:
            # Получаем информацию о видео
            probe = ffmpeg.probe(video_path)
            
            # Извлекаем нужные данные
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            
            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate']),
                'audio_sample_rate': int(audio_info['sample_rate']),
                'audio_channels': int(audio_info['channels'])
            }
            
        except ffmpeg.Error as e:
            logger.error(f"Ошибка при получении информации о видео: {e.stderr.decode()}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при получении информации о видео: {e}")
            return None
    
    def merge_audio_with_video(self, video_path, audio_path, output_path, 
                               background_volume=0.1, original_fps=None):
        """
        Наложение синтезированной речи на видео
        
        Args:
            video_path (str): Путь к оригинальному видео
            audio_path (str): Путь к синтезированной аудиодорожке
            output_path (str): Путь для сохранения результата
            background_volume (float): Громкость оригинального аудио (0.0-1.0)
            original_fps (float, optional): Частота кадров для выходного видео
            
        Returns:
            str: Путь к сохраненному видеофайлу
        """
        logger.info("Начинаем объединение аудио с видео...")
        
        try:
            # Создаем временную директорию для обработки
            with tempfile.TemporaryDirectory() as temp_dir:
                # Извлекаем оригинальную аудиодорожку из видео
                original_audio_path = os.path.join(temp_dir, "original_audio.wav")
                try:
                    # Извлекаем аудио
                    (
                        ffmpeg
                        .input(video_path)
                        .output(original_audio_path, acodec='pcm_s16le')
                        .global_args('-y')
                        .run(quiet=True, overwrite_output=True)
                    )
                    logger.info(f"Извлечена оригинальная аудиодорожка: {original_audio_path}")
                    
                    # Проверяем наличие файла
                    if not os.path.exists(original_audio_path) or os.path.getsize(original_audio_path) == 0:
                        logger.warning("Оригинальная аудиодорожка пуста или не создана. Возможно, видео не содержит аудио.")
                        has_original_audio = False
                    else:
                        has_original_audio = True
                except Exception as e:
                    logger.warning(f"Не удалось извлечь оригинальную аудиодорожку: {e}")
                    has_original_audio = False
                
                # Получаем информацию о видео для сохранения fps и других параметров
                try:
                    probe = ffmpeg.probe(video_path)
                    video_stream = next((stream for stream in probe['streams'] 
                                        if stream['codec_type'] == 'video'), None)
                    
                    # Вычисляем FPS
                    if video_stream and 'r_frame_rate' in video_stream:
                        fps_parts = video_stream['r_frame_rate'].split('/')
                        fps = float(fps_parts[0]) / float(fps_parts[1] if len(fps_parts) > 1 else 1)
                    else:
                        fps = original_fps if original_fps is not None else 25
                except Exception as e:
                    logger.warning(f"Не удалось получить информацию о видео: {e}")
                    fps = original_fps if original_fps is not None else 25
                
                # Если нужно сохранить оригинальный звук в фоне
                final_audio_path = os.path.join(temp_dir, "final_audio.wav")
                if has_original_audio and background_volume > 0:
                    try:
                        # Загружаем аудио с помощью pydub для регулировки громкости
                        original_audio = AudioSegment.from_file(original_audio_path)
                        dubbed_audio = AudioSegment.from_file(audio_path)
                        
                        # Регулируем громкость оригинального аудио
                        background = original_audio - (20 - int(background_volume * 20))  # Конвертируем в dB
                        
                        # Накладываем дубляж на тихий оригинал
                        final_audio = background.overlay(dubbed_audio)
                        
                        # Сохраняем результат
                        final_audio.export(final_audio_path, format="wav")
                        logger.info(f"Создана композитная аудиодорожка с фоновым звуком: {final_audio_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при создании композитной аудиодорожки: {e}")
                        # В случае ошибки используем только дубляж
                        final_audio_path = audio_path
                else:
                    # Если нет оригинальной дорожки или фон не нужен, используем только дубляж
                    final_audio_path = audio_path
                
                # Создаем итоговое видео с новой аудиодорожкой
                try:
                    # Создаем отдельные потоки для видео и аудио
                    video_stream = ffmpeg.input(video_path)
                    audio_stream = ffmpeg.input(final_audio_path)
                    
                    # Объединяем их в выходной поток
                    (
                        ffmpeg
                        .output(
                            video_stream, audio_stream,
                            output_path,
                            vcodec=self.codec,
                            **{'b:v': self.bitrate},
                            acodec='aac',
                            **{'b:a': '192k'},
                            **{'map': ['0:v', '1:a']},
                            r=fps
                        )
                        .global_args('-y')
                        .run(quiet=True, overwrite_output=True)
                    )
                    
                    # Проверяем результат
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logger.info(f"Объединение завершено. Финальное видео сохранено: {output_path}")
                        return output_path
                    else:
                        logger.error(f"Не удалось создать итоговое видео или файл пуст: {output_path}")
                        raise RuntimeError("Создан пустой выходной файл")
                except ffmpeg.Error as e:
                    logger.error(f"Ошибка ffmpeg при создании видео: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
                    raise RuntimeError(f"Ошибка при создании видео: {str(e)}")
                
        except Exception as e:
            logger.error(f"Общая ошибка при объединении аудио с видео: {e}")
            raise 