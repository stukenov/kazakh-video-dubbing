"""
Модуль для управления синтезом речи
"""
import os
import logging
import asyncio
import random
from .tts_services import YandexTTSService, EdgeTTSService
from utils.config import Config

logger = logging.getLogger(__name__)

class TTSManager:
    """
    Класс для управления синтезом речи с поддержкой различных TTS движков
    """
    def __init__(self, config=None):
        """
        Инициализация менеджера TTS с поддержкой Edge TTS и Yandex SpeechKit
        :param config: Конфигурация с настройками TTS
        """
        # Если конфигурация не передана, создаем новую
        self.config = config or Config()
        self.tts_config = self.config.TTS
        
        # Кэш для последнего использованного голоса
        self._last_used_voice = {}
        
        # Инициализируем сервисы TTS
        self._init_tts_services()
        
        logger.info(f"Инициализирован менеджер TTS (используется {self.tts_config['service']})")
    
    def _init_tts_services(self):
        """
        Инициализация сервисов TTS на основе конфигурации
        """
        # Инициализируем Yandex TTS
        try:
            self.yandex_service = YandexTTSService(
                api_key=self.config.yandex_speechkit_api_key
            )
        except ValueError as e:
            logger.warning(f"Не удалось инициализировать Yandex TTS: {str(e)}")
            self.yandex_service = None
        
        # Инициализируем Edge TTS
        edge_config = self.tts_config.get('edge', {})
        self.edge_service = EdgeTTSService(
            rate=edge_config.get('rate', '+0%'),
            volume=edge_config.get('volume', '+0%'),
            pitch=edge_config.get('pitch', '+0Hz')
        )
    
    def _get_voice_for_service(self, language: str, gender: str) -> str:
        """
        Получает имя голоса для выбранного сервиса с учетом разнообразия голосов
        
        Args:
            language (str): Код языка
            gender (str): Пол говорящего ('male' или 'female')
            
        Returns:
            str: Имя голоса
        """
        try:
            voices = self.tts_config['voices'][language]
            
            # Для Edge TTS просто берем голос по полу
            if self.tts_config['service'] == 'edge':
                return voices[gender]
            
            # Для Yandex SpeechKit
            elif self.tts_config['service'] == 'yandex':
                available_voices = voices[gender]
                
                # Если это словарь с именами голосов
                if isinstance(available_voices, dict):
                    # Если это тот же язык и пол, что и в прошлый раз
                    if (self._last_used_voice.get('language') == language and 
                        self._last_used_voice.get('gender') == gender and 
                        self._last_used_voice.get('voice_name')):
                        # Пытаемся использовать другой голос
                        current_voice_name = self._last_used_voice['voice_name']
                        voice_names = list(available_voices.keys())
                        if len(voice_names) > 1:
                            # Исключаем текущий голос из выбора
                            voice_names.remove(current_voice_name)
                            # Выбираем случайный голос из оставшихся
                            next_voice_name = random.choice(voice_names)
                            voice_id = available_voices[next_voice_name]
                            self._last_used_voice = {
                                'language': language,
                                'gender': gender,
                                'voice_name': next_voice_name
                            }
                            logger.info(f"Выбран новый голос {next_voice_name} для разнообразия")
                            return voice_id
                    
                    # Если первый раз или другой язык/пол
                    voice_name = random.choice(list(available_voices.keys()))
                    voice_id = available_voices[voice_name]
                    self._last_used_voice = {
                        'language': language,
                        'gender': gender,
                        'voice_name': voice_name
                    }
                    logger.info(f"Выбран голос {voice_name}")
                    return voice_id
                
                # Если это просто строка с ID голоса
                return available_voices
            
        except KeyError:
            logger.warning(f"Голос не найден для языка {language} и пола {gender}, используем запасной вариант")
            # Возвращаем первый доступный голос для этого языка
            try:
                if self.tts_config['service'] == 'yandex':
                    # Для Yandex берем первый голос из любого пола
                    for gender_voices in voices.values():
                        if isinstance(gender_voices, dict):
                            first_voice = list(gender_voices.values())[0]
                            return first_voice
                        return gender_voices
                else:
                    # Для других сервисов берем первый доступный голос
                    return list(voices.values())[0]
            except (KeyError, IndexError):
                logger.error(f"Не удалось найти запасной голос для языка {language}")
                raise ValueError(f"Нет доступных голосов для языка {language}")
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = 'kk',
        voice: str = None,
        gender: str = None
    ) -> bool:
        """
        Синтез речи с использованием выбранного движка
        
        Args:
            text (str): Текст для синтеза
            output_path (str): Путь для сохранения аудио
            language (str): Язык синтеза
            voice (str, optional): Имя голоса (если не указан, выбирается из конфигурации)
            gender (str, optional): Пол говорящего (если не указан, берется из конфигурации)
            
        Returns:
            bool: True если синтез успешен, False в противном случае
        """
        try:
            # Создаем директорию для выходного файла если её нет
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Определяем параметры синтеза
            gender = gender or self.tts_config['voice_gender']
            voice = voice or self._get_voice_for_service(language, gender)
            
            # Получаем сервис и запускаем синтез
            if self.tts_config['service'] == 'yandex' and self.yandex_service:
                return asyncio.create_task(
                    self.yandex_service.synthesize(text, output_path, language, voice)
                )
            else:
                return asyncio.create_task(
                    self.edge_service.synthesize(text, output_path, language, voice)
                )
                
        except Exception as e:
            logger.error(f"Ошибка при синтезе речи: {e}")
            return False 