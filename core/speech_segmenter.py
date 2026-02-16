"""
Модуль для сегментации речи и определения пола говорящего
"""
import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import torch
import torchaudio
import whisper
from pydub import AudioSegment
import librosa
from scipy.stats import mode
from scipy import signal
import torch.nn.functional as F
import noisereduce as nr
import tensorflow as tf
import tensorflow_hub as hub

from models.speech_segment import SpeechSegment

logger = logging.getLogger(__name__)

@dataclass
class SpeechSegment:
    """Класс для хранения информации о сегменте речи"""
    text: str                # Текст сегмента
    start_time: float        # Время начала в секундах
    end_time: float          # Время окончания в секундах
    speaker_gender: str      # Пол говорящего ('male', 'female', 'unknown')
    gender_confidence: float = 0.0  # Уверенность в определении пола (0-1)
    translated_text: str = None  # Переведенный текст (заполняется позже)
    audio_path: str = None   # Путь к аудиофайлу сегмента (заполняется позже)

class SpeechSegmenter:
    """
    Класс для сегментации речи, определения временных меток и пола говорящего
    """
    def __init__(self, model_name: str = "medium", language: str = "en"):
        """
        Инициализация сегментатора речи
        
        Args:
            model_name (str): Название модели Whisper
            language (str): Язык для распознавания речи
        """
        self.model = whisper.load_model(model_name)
        self.language = language
        logger.info(f"Инициализирован сегментатор речи с моделью {model_name} для языка {language}")
        
        # Модель для определения пола
        self.gender_model = None  # YAMNet для анализа аудио характеристик
        
        # Параметры для улучшения качества голоса
        self.voice_enhancement_params = {
            'noise_reduce_kwargs': {
                'stationary': True,
                'prop_decrease': 0.95,
                'n_jobs': -1
            },
            'bandpass_kwargs': {
                'lowcut': 80.0,   # Нижняя частота для голоса
                'highcut': 3400.0, # Верхняя частота для голоса
                'order': 5
            }
        }
    
    def _ensure_models_loaded(self):
        """Загрузка всех необходимых моделей"""
        if self.model is None:
            logger.info(f"Загрузка модели Whisper {self.model_size}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(self.model_size, device=device)
            logger.info(f"Модель Whisper загружена, используется {device}")
        
        if self.gender_model is None:
            logger.info("Загрузка модели классификации пола...")
            try:
                # Загрузка YAMNet через TensorFlow Hub
                self.gender_model = hub.load('https://tfhub.dev/google/yamnet/1')
                logger.info("YAMNet модель загружена")
            except Exception as e:
                logger.warning(f"Ошибка загрузки YAMNet модели: {e}")

    def _enhance_voice(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Улучшение качества голоса в аудио
        
        Args:
            audio_data: аудио данные
            sr: частота дискретизации
            
        Returns:
            np.ndarray: улучшенные аудио данные
        """
        try:
            # 1. Нормализация амплитуды
            audio_data = librosa.util.normalize(audio_data)
            
            # 2. Применение полосового фильтра для выделения голосового диапазона
            bp_params = self.voice_enhancement_params['bandpass_kwargs']
            b, a = signal.butter(bp_params['order'], 
                               [bp_params['lowcut'] / (sr/2), 
                                bp_params['highcut'] / (sr/2)], 
                               btype='band')
            audio_data = signal.filtfilt(b, a, audio_data)
            
            # 3. Шумоподавление
            audio_data = nr.reduce_noise(y=audio_data,
                                       sr=sr,
                                       **self.voice_enhancement_params['noise_reduce_kwargs'])
            
            # 4. Усиление речевых частот
            emphasized = librosa.effects.preemphasis(audio_data)
            
            # 5. Финальная нормализация
            audio_data = librosa.util.normalize(emphasized)
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Ошибка при улучшении качества голоса: {str(e)}")
            return audio_data

    def _estimate_speaker_gender(self, audio_path: str, start_time: float, end_time: float, 
                               sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Определение пола говорящего с использованием YAMNet
        
        Args:
            audio_path: путь к аудиофайлу
            start_time: время начала сегмента
            end_time: время окончания сегмента
            sample_rate: частота дискретизации
            
        Returns:
            Tuple[str, float]: кортеж (определенный пол, уверенность)
        """
        try:
            self._ensure_models_loaded()
            
            # Загружаем аудио
            audio_data, sr = librosa.load(audio_path, sr=sample_rate, 
                                        offset=start_time, 
                                        duration=end_time-start_time)
            
            # Улучшаем качество голоса
            enhanced_audio = self._enhance_voice(audio_data, sr)
            
            # Подготовка входных данных для YAMNet
            if sr != 16000:
                enhanced_audio = librosa.resample(enhanced_audio, orig_sr=sr, target_sr=16000)
            
            # Нормализация
            enhanced_audio = librosa.util.normalize(enhanced_audio)
            
            # Получение предсказаний от YAMNet
            scores, embeddings, spectrogram = self.gender_model(enhanced_audio)
            scores = scores.numpy()
            
            # Получаем средние значения по всем фреймам для каждого класса
            mean_scores = np.mean(scores, axis=0)
            
            # Индексы для мужского и женского голоса в YAMNet
            # Индекс 0: "Speech" - общий класс для речи
            # Индекс 1: "Male speech, man speaking" - мужской голос
            # Индекс 2: "Female speech, woman speaking" - женский голос
            male_index = 1
            female_index = 2
            
            # Получаем оценки для мужского и женского голоса
            male_prob = mean_scores[male_index]
            female_prob = mean_scores[female_index]
            
            # Для отладки логируем оценки для первых 5 классов
            logger.debug(f"YAMNet оценки для аудио [{start_time:.2f}s - {end_time:.2f}s]:")
            logger.debug(f"  Класс 0 (Речь): {mean_scores[0]:.4f}")
            logger.debug(f"  Класс 1 (Мужской голос): {male_prob:.4f}")
            logger.debug(f"  Класс 2 (Женский голос): {female_prob:.4f}")
            logger.debug(f"  Класс 3 (Детский голос): {mean_scores[3]:.4f}")
            logger.debug(f"  Класс 4 (Разговор): {mean_scores[4]:.4f}")
            
            # Определяем итоговый результат
            if max(male_prob, female_prob) < 0.05:  # Низкий порог для большей чувствительности
                # Если значения очень низкие, но есть речь, используем вероятностный подход
                if mean_scores[0] > 0.3:  # Если есть речь с высокой вероятностью
                    # Если между мужским и женским голосом небольшая разница
                    if abs(male_prob - female_prob) < 0.02:
                        # Сравниваем также с детским голосом
                        if mean_scores[3] > max(male_prob, female_prob):
                            # Если детский голос более выражен, скорее всего это женский
                            return 'female', 0.5
                        # Иначе вероятно мужской (по умолчанию)
                        return 'male', 0.5
                    
                    # Даже при низких значениях выбираем наибольшее
                    if male_prob > female_prob:
                        return 'male', male_prob * 2  # Увеличиваем уверенность
                    else:
                        return 'female', female_prob * 2  # Увеличиваем уверенность
                
                # Если речи нет или неуверены, возвращаем unknown
                return 'unknown', 0.0
            
            # Если значения уверенные, просто выбираем наибольшее
            if male_prob > female_prob:
                confidence = min(0.99, male_prob)
                logger.debug(f"Определен мужской пол с уверенностью {confidence:.2f}")
                return 'male', confidence
            else:
                confidence = min(0.99, female_prob)
                logger.debug(f"Определен женский пол с уверенностью {confidence:.2f}")
                return 'female', confidence
            
        except Exception as e:
            logger.error(f"Ошибка при определении пола говорящего: {e}")
            return 'unknown', 0.0
    
    def _extract_segments_with_timestamps(self, audio_path):
        """
        Извлечение сегментов речи с временными метками
        
        Args:
            audio_path (str): Путь к аудиофайлу
            
        Returns:
            List[dict]: Список сегментов с текстом и временными метками
        """
        self._ensure_models_loaded()
        
        # Транскрибируем аудио с временными метками
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,  # Включаем временные метки для слов
            verbose=False
        )
        
        # Извлекаем сегменты
        segments = []
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            if text:  # Пропускаем пустые сегменты
                segments.append({
                    "text": text,
                    "start": start,
                    "end": end
                })
        
        return segments
    
    def _split_audio_by_segments(self, audio_path, output_dir, segments):
        """
        Разделение аудиофайла на сегменты
        
        Args:
            audio_path (str): Путь к исходному аудиофайлу
            output_dir (str): Директория для сохранения сегментов
            segments (List[SpeechSegment]): Список сегментов
            
        Returns:
            List[SpeechSegment]: Обновленный список сегментов с путями к аудиофайлам
        """
        # Загрузка аудиофайла
        audio = AudioSegment.from_file(audio_path)
        
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Разделяем аудио и сохраняем каждый сегмент
        for i, segment in enumerate(segments):
            start_ms = int(segment.start_time * 1000)
            end_ms = int(segment.end_time * 1000)
            
            # Вырезаем сегмент
            segment_audio = audio[start_ms:end_ms]
            
            # Создаем имя файла для сегмента
            segment_filename = f"segment_{i:04d}_{start_ms:08d}_{end_ms:08d}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # Сохраняем сегмент
            segment_audio.export(segment_path, format="wav")
            
            # Обновляем путь в сегменте
            segment.audio_path = segment_path
        
        return segments
    
    def segment_audio(self, audio_path, output_dir):
        """
        Сегментация аудио с определением времени и пола говорящего
        
        Args:
            audio_path (str): Путь к аудиофайлу
            output_dir (str): Директория для сохранения сегментов
            
        Returns:
            List[SpeechSegment]: Список объектов сегментов речи
        """
        logger.info(f"Начинаем сегментацию аудио: {audio_path}")
        
        # Извлекаем сегменты с временными метками
        segments_data = self._extract_segments_with_timestamps(audio_path)
        
        # Создаем объекты SpeechSegment и определяем пол говорящего
        speech_segments = []
        for i, segment_data in enumerate(segments_data):
            # Определяем пол говорящего
            speaker_gender, confidence = self._estimate_speaker_gender(
                audio_path, segment_data["start"], segment_data["end"]
            )
            
            # Создаем объект сегмента
            segment = SpeechSegment(
                text=segment_data["text"],
                start_time=segment_data["start"],
                end_time=segment_data["end"],
                speaker_gender=speaker_gender,
                gender_confidence=confidence
            )
            
            speech_segments.append(segment)
            
            logger.info(
                f"Сегмент {i+1}/{len(segments_data)}: "
                f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                f"Пол: {segment.speaker_gender}, "
                f"Текст: {segment.text[:50]}{'...' if len(segment.text) > 50 else ''}"
            )
        
        # Разделяем аудио по сегментам
        speech_segments = self._split_audio_by_segments(audio_path, output_dir, speech_segments)
        
        logger.info(f"Сегментация завершена. Получено {len(speech_segments)} сегментов")
        return speech_segments 