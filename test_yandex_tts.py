"""
Тестовый скрипт для проверки Yandex TTS
"""
import os
import asyncio
import logging
from services.tts_services import YandexTTSService
from utils.config import Config

# Настраиваем подробное логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_yandex_tts():
    """Тестирование Yandex TTS с подробной проверкой"""
    try:
        config = Config()
        
        # Проверяем наличие API ключа
        if not config.yandex_speechkit_api_key:
            logger.error("API ключ Yandex SpeechKit не найден в конфигурации")
            return False
            
        logger.info(f"API ключ найден, длина: {len(config.yandex_speechkit_api_key)} символов")
        
        # Создаем сервис
        service = YandexTTSService(api_key=config.yandex_speechkit_api_key)
        
        # Тестовые тексты на казахском разной длины
        test_texts = [
            "Сәлем!",  # Короткий текст
            "Бұл қазақ тілінде сынақ мәтін.",  # Средний текст
            "Бұл ұзағырақ сынақ мәтін. Біз дыбыс синтезін тексереміз. Yandex SpeechKit қызметі жұмыс істейді."  # Длинный текст
        ]
        
        # Создаем директорию для тестов
        test_dir = "test_output/yandex_test"
        os.makedirs(test_dir, exist_ok=True)
        logger.info(f"Создана директория для тестов: {test_dir}")
        
        # Доступные голоса для казахского языка в Yandex SpeechKit
        voices = {
            'male': 'madi',  # Мужской голос
            'female': 'amira'  # Женский голос
        }
        
        # Тестируем синтез для каждого текста с разными голосами
        for voice_type, voice_name in voices.items():
            logger.info(f"\nТестирование голоса: {voice_type} ({voice_name})")
            
            for i, text in enumerate(test_texts, 1):
                logger.info(f"\nТест #{i}")
                logger.info(f"Текст для синтеза ({len(text)} символов): {text}")
                
                output_path = os.path.join(test_dir, f"test_{voice_type}_{i}.wav")
                
                # Если файл существует, удаляем его
                if os.path.exists(output_path):
                    os.remove(output_path)
                    logger.info(f"Удален существующий файл: {output_path}")
                
                # Синтезируем речь
                logger.info(f"Начинаем синтез с голосом {voice_name}...")
                result = await service.synthesize(
                    text=text,
                    output_path=output_path,
                    language="kk-KK",
                    voice=voice_name
                )
                
                # Проверяем результат
                if result:
                    if os.path.exists(output_path):
                        size = os.path.getsize(output_path)
                        logger.info(f"Файл успешно создан: {output_path}")
                        logger.info(f"Размер файла: {size} байт")
                        
                        if size == 0:
                            logger.error("Файл создан, но пустой!")
                        else:
                            logger.info("Тест успешен!")
                    else:
                        logger.error(f"Файл не найден несмотря на успешный результат: {output_path}")
                else:
                    logger.error("Синтез завершился с ошибкой")
        
        logger.info("\nВсе тесты завершены!")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении теста: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(test_yandex_tts()) 