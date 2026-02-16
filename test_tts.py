"""
Тестовый скрипт для проверки синтеза речи
"""
import os
import asyncio
import logging
from services.tts_services import YandexTTSService
from utils.config import Config

# Настраиваем логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_yandex_tts():
    """Тестирование Yandex TTS"""
    config = Config()
    
    # Создаем сервис
    service = YandexTTSService(api_key=config.yandex_speechkit_api_key)
    
    # Тестовый текст на казахском
    test_text = "Сәлеметсіз бе! Бұл сынақ мәтіні."
    
    # Создаем директорию для тестов если её нет
    os.makedirs("test_output", exist_ok=True)
    
    # Пробуем синтезировать речь
    result = await service.synthesize(
        text=test_text,
        output_path="test_output/test_yandex.wav",
        language="kk-KK"
    )
    
    if result:
        print("Тест успешно завершен! Проверьте файл test_output/test_yandex.wav")
    else:
        print("Тест не удался!")

if __name__ == "__main__":
    asyncio.run(test_yandex_tts()) 