"""
Модуль с базовыми классами для сервисов TTS
"""
import abc
import os
import logging
import requests
import edge_tts
import asyncio
from typing import Optional
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class TextToSpeechService(abc.ABC):
    """
    Абстрактный класс для сервисов преобразования текста в речь.
    Этот класс определяет общий интерфейс для различных TTS сервисов.
    """
    
    @abc.abstractmethod
    async def synthesize(self, text: str, output_path: str, language: str = None, voice: str = None) -> bool:
        """
        Преобразует текст в речь и сохраняет результат в файл.
        
        Args:
            text (str): Текст для преобразования в речь
            output_path (str): Путь для сохранения аудиофайла
            language (str, optional): Язык синтеза речи
            voice (str, optional): Голос для синтеза
            
        Returns:
            bool: True если синтез успешен, False в противном случае
        """
        pass


class YandexTTSService(TextToSpeechService):
    """
    Реализация сервиса преобразования текста в речь с использованием Yandex SpeechKit.
    Позволяет синтезировать речь на казахском языке с высоким качеством.
    """
    
    # Константы для работы с Yandex SpeechKit API
    SPEECHKIT_URL = 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'
    DEFAULT_LANGUAGE = "kk-KK"  # Казахский язык по умолчанию
    
    def __init__(self, api_key: str):
        """
        Инициализирует сервис с API ключом.
        
        Args:
            api_key (str): API ключ Yandex SpeechKit (сервисный аккаунт)
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API ключ Yandex SpeechKit не найден")
    
    async def synthesize(self, text: str, output_path: str, language: str = None, voice: str = None) -> bool:
        """
        Преобразует текст в речь используя Yandex SpeechKit.
        
        Args:
            text (str): Текст для преобразования
            output_path (str): Путь для сохранения аудиофайла
            language (str, optional): Язык синтеза речи (по умолчанию kk-KK)
            voice (str, optional): Голос для синтеза
            
        Returns:
            bool: True если синтез успешен, False в противном случае
        """
        try:
            # Подготавливаем заголовки и данные для запроса
            headers = {'Authorization': f'Api-Key {self.api_key}'}
            
            # Используем OGG вместо LPCM для более надежной работы
            data = {
                'text': text,
                'lang': language or self.DEFAULT_LANGUAGE,
                'voice': voice,
                'format': 'oggopus',  # Используем OGG формат вместо LPCM
                'sampleRateHertz': '48000',  # Высокое качество сэмплирования
                'speed': 1.0  # Нормальная скорость речи
            }
            
            logger.info(f"Отправляем запрос в Yandex TTS для текста длиной {len(text)} символов")
            logger.debug(f"Параметры запроса: язык={data['lang']}, голос={voice}, формат={data['format']}")
            
            # Создаем временный файл для OGG
            temp_ogg = output_path + '.tmp.ogg'
            
            try:
                # Создаем сессию для асинхронных запросов
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.SPEECHKIT_URL, headers=headers, data=data) as response:
                        # Проверяем успешность запроса
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Ошибка синтеза речи: {response.status}, {error_text}")
                            return False
                        
                        # Создаем директорию для выходного файла если её нет
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Сохраняем полученные данные во временный файл
                        content = await response.read()
                        
                        if len(content) == 0:
                            logger.error("Получен пустой ответ от Yandex TTS")
                            return False
                            
                        logger.debug(f"Получено {len(content)} байт аудио данных")
                        
                        with open(temp_ogg, 'wb') as f:
                            f.write(content)
                        
                        if not os.path.exists(temp_ogg) or os.path.getsize(temp_ogg) == 0:
                            logger.error("Временный OGG файл не создан или пуст")
                            return False
                            
                        logger.debug(f"Сохранен временный OGG файл размером {os.path.getsize(temp_ogg)} байт")
                        
                        try:
                            # Загружаем OGG файл через pydub
                            audio = AudioSegment.from_ogg(temp_ogg)
                            
                            # Проверяем параметры аудио
                            logger.debug(f"Параметры аудио: частота={audio.frame_rate}Hz, "
                                       f"каналы={audio.channels}, битность={audio.sample_width*8}bit")
                            
                            # Конвертируем в нужный формат
                            audio = audio.set_frame_rate(48000)  # 48kHz
                            audio = audio.set_channels(1)        # Mono
                            audio = audio.set_sample_width(2)    # 16-bit
                            
                            # Экспортируем в WAV
                            audio.export(output_path, format='wav')
                            
                            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                                logger.error("Итоговый WAV файл не создан или пуст")
                                return False
                                
                            logger.info(f"Yandex TTS: Аудио успешно сохранено в {output_path} "
                                      f"(размер: {os.path.getsize(output_path)} байт)")
                            return True
                            
                        except Exception as e:
                            logger.error(f"Ошибка при конвертации аудио: {str(e)}")
                            return False
                        
            finally:
                # Удаляем временный файл
                try:
                    if os.path.exists(temp_ogg):
                        os.remove(temp_ogg)
                        logger.debug("Временный OGG файл удален")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {temp_ogg}: {e}")
            
        except Exception as e:
            logger.error(f"Ошибка при синтезе речи через Yandex TTS: {str(e)}")
            return False


class EdgeTTSService(TextToSpeechService):
    """
    Реализация сервиса преобразования текста в речь с использованием Edge TTS.
    """
    
    def __init__(self, rate: str = '+0%', volume: str = '+0%', pitch: str = '+0Hz'):
        """
        Инициализирует сервис Edge TTS.
        
        Args:
            rate (str): Скорость речи
            volume (str): Громкость
            pitch (str): Высота тона
        """
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.default_voice = 'kk-KZ-AigulNeural'  # Казахский женский голос по умолчанию
    
    async def synthesize(self, text: str, output_path: str, language: str = None, voice: str = None) -> bool:
        """
        Преобразует текст в речь используя Edge TTS.
        
        Args:
            text (str): Текст для преобразования
            output_path (str): Путь для сохранения аудиофайла
            language (str, optional): Язык синтеза речи (не используется в Edge TTS)
            voice (str): Голос для синтеза
            
        Returns:
            bool: True если синтез успешен, False в противном случае
        """
        try:
            # Используем переданный голос или голос по умолчанию
            voice = voice or self.default_voice
            
            if not voice:
                logger.error("Edge TTS: Не указан голос для синтеза")
                return False
            
            # Создаем директорию для выходного файла если её нет
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Создаем временный файл для MP3
            temp_mp3 = output_path + '.tmp.mp3'
            
            # Настраиваем параметры синтеза
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )
            
            # Синтезируем речь во временный MP3 файл
            await communicate.save(temp_mp3)
            
            # Проверяем результат синтеза
            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                logger.error("Edge TTS: Не удалось синтезировать речь или файл пуст")
                return False
            
            try:
                # Конвертируем MP3 в WAV с нужными параметрами
                audio = AudioSegment.from_mp3(temp_mp3)
                # Устанавливаем параметры WAV файла
                audio = audio.set_frame_rate(48000)  # 48kHz
                audio = audio.set_channels(1)        # Mono
                audio = audio.set_sample_width(2)    # 16-bit
                
                # Экспортируем в WAV
                audio.export(output_path, format='wav')
                
                logger.info(f"Edge TTS: Речь успешно синтезирована и конвертирована в WAV: {output_path}")
                return True
                
            finally:
                # Удаляем временный MP3 файл
                try:
                    os.remove(temp_mp3)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {temp_mp3}: {e}")
                
        except Exception as e:
            logger.error(f"Ошибка при синтезе речи через Edge TTS: {str(e)}")
            return False 