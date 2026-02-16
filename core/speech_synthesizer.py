"""
Модуль для синтеза речи на казахском языке с помощью Edge TTS
"""
import logging
import asyncio
import edge_tts
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class EdgeTTSSynthesizer:
    """
    Класс для синтеза речи с использованием Microsoft Edge TTS
    """
    def __init__(self, language='kk', voice_gender='female', speed=1.0, pitch=1.0):
        """
        Инициализация синтезатора речи
        
        Args:
            language (str): Код языка (kk для казахского)
            voice_gender (str): Пол голоса (female/male)
            speed (float): Скорость речи (1.0 - нормальная)
            pitch (float): Высота голоса (1.0 - нормальная)
        """
        self.language = language
        self.voice_gender = voice_gender
        self.speed = speed
        self.pitch = pitch
        self.voices = None
        logger.info(f"Инициализирован синтезатор речи Edge TTS для языка {language}")
    
    async def _get_voices(self):
        """
        Получает список доступных голосов Edge TTS
        
        Returns:
            list: Список доступных голосов
        """
        voices = await edge_tts.list_voices()
        return voices
    
    def _get_voice(self, loop=None):
        """
        Возвращает подходящий голос для Edge TTS
        
        Args:
            loop (asyncio.AbstractEventLoop, optional): Цикл событий asyncio
            
        Returns:
            str: Идентификатор голоса
        """
        # Получаем список голосов, если еще не получен
        if self.voices is None:
            if loop is None:
                loop = asyncio.get_event_loop()
            self.voices = loop.run_until_complete(self._get_voices())
        
        # Ищем голоса для указанного языка
        target_voices = [v for v in self.voices if v.get("Locale", "").startswith(self.language)]
        
        # Если голоса найдены, выбираем по полу
        if target_voices:
            if self.voice_gender == 'female':
                female_voices = [v for v in target_voices if "Female" in v.get("Gender", "")]
                if female_voices:
                    return female_voices[0]["ShortName"]
                # Если женских нет, возвращаем любой
                return target_voices[0]["ShortName"]
            else:  # male voice
                male_voices = [v for v in target_voices if "Male" in v.get("Gender", "")]
                if male_voices:
                    return male_voices[0]["ShortName"]
                # Если мужских нет, возвращаем любой
                return target_voices[0]["ShortName"]
        
        # Если голосов для указанного языка нет, ищем русский (ближайший)
        russian_voices = [v for v in self.voices if v.get("Locale", "").startswith("ru")]
        if russian_voices:
            if self.voice_gender == 'female':
                female_voices = [v for v in russian_voices if "Female" in v.get("Gender", "")]
                if female_voices:
                    return female_voices[0]["ShortName"]
                return russian_voices[0]["ShortName"]
            else:  # male voice
                male_voices = [v for v in russian_voices if "Male" in v.get("Gender", "")]
                if male_voices:
                    return male_voices[0]["ShortName"]
                return russian_voices[0]["ShortName"]
        
        # Если нет ни целевого языка, ни русских, используем первый доступный
        return self.voices[0]["ShortName"] if self.voices else "ru-RU-SvetlanaNeural"
    
    async def _synthesize_async(self, text, output_path):
        """
        Асинхронная функция для синтеза речи
        
        Args:
            text (str): Текст для синтеза
            output_path (str): Путь для сохранения аудиофайла
            
        Returns:
            str: Путь к сохраненному аудиофайлу
        """
        voice = self._get_voice()
        logger.info(f"Используем голос Edge TTS: {voice}")
        
        # Форматирование параметров для Edge TTS
        # Скорость речи
        if self.speed == 1.0:
            rate = "+0%"
        elif self.speed > 1.0:
            # Если скорость выше 1.0, используем положительный процент
            percent = int((self.speed - 1.0) * 100)
            rate = f"+{percent}%"
        else:
            # Если скорость ниже 1.0, используем отрицательный процент
            percent = int((1.0 - self.speed) * 100)
            rate = f"-{percent}%"
        
        # Громкость
        volume = "+0%"  # Стандартная громкость
        
        # Высота тона
        if self.pitch == 1.0:
            pitch = "+0Hz"  # Стандартная высота тона
        elif self.pitch > 1.0:
            # Если высота больше 1.0, используем положительный сдвиг
            hz_shift = int((self.pitch - 1.0) * 50)
            pitch = f"+{hz_shift}Hz"
        else:
            # Если высота меньше 1.0, используем отрицательный сдвиг
            hz_shift = int((1.0 - self.pitch) * 50)
            pitch = f"-{hz_shift}Hz"
        
        logger.info(f"Edge TTS параметры: голос={voice}, скорость={rate}, высота={pitch}")
        
        try:
            # Синтезируем речь
            communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
            await communicate.save(output_path)
            return output_path
        except Exception as e:
            # В случае ошибки пробуем с базовыми настройками
            logger.error(f"Ошибка при синтезе речи: {e}")
            logger.info("Попытка с базовыми настройками Edge TTS")
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_path)
                return output_path
            except Exception as e2:
                logger.error(f"Вторая ошибка при синтезе речи: {e2}")
                raise e2
    
    def synthesize(self, text, output_path, loop=None):
        """
        Синтез речи из текста и сохранение в файл
        
        Args:
            text (str): Текст для синтеза
            output_path (str): Путь для сохранения аудиофайла
            loop (asyncio.AbstractEventLoop, optional): Цикл событий asyncio
            
        Returns:
            str: Путь к сохраненному аудиофайлу
        """
        logger.info("Начинаем синтез речи с помощью Edge TTS...")
        
        try:
            # Используем предоставленный цикл событий или создаем новый
            if loop is None:
                loop = asyncio.get_event_loop()
            
            # Запускаем синтез речи
            output_path = loop.run_until_complete(self._synthesize_async(text, output_path))
            logger.info(f"Синтез речи завершен. Результат сохранен: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при синтезе речи: {e}")
            logger.warning("Создаем пустой аудиофайл вместо синтезированной речи")
            # Создаем пустой аудиофайл для продолжения обработки
            silent_audio = AudioSegment.silent(duration=5000)  # 5 секунд тишины
            silent_audio.export(output_path, format="mp3")
            logger.info(f"Создан пустой аудиофайл: {output_path}")
            return output_path 