import requests
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO
from telegram import Bot
from telegram.ext import Application, ContextTypes
from telegram.error import TelegramError
import asyncio
import time

# Ваш API-ключ NASA
api_key = ''

# Ваш токен Telegram-бота
telegram_token = 'YOUR_TELEGRAM_BOT_TOKEN'

# ID вашего чата (личка, либо группа)
chat_id = 'YOUR_CHAT_ID'

# Базовый URL для API Mars Rover Photos
api_url = 'https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos'

# Инициализация моделей YOLOv8
rfr = YOLO('rfr.pt')  # Модель для поиска окаменелостей
mollusc = YOLO('mollusc.pt')  # Модель для поиска моллюсков (пока закомментирована)

# Функция для отправки изображений в Telegram
async def send_image_to_telegram(img, chat_id, bot, label):
    try:
        # Преобразуем изображение в формат для отправки через Telegram
        bio = BytesIO()
        bio.name = 'image.jpg'  # Имя файла для Telegram
        img.save(bio, 'JPEG')
        bio.seek(0)

        # Отправляем изображение в Telegram с подписью
        await bot.send_photo(chat_id=chat_id, photo=bio, caption=label)
        print("Изображение отправлено в Telegram с подписью.")
    except TelegramError as e:
        print(f"Ошибка отправки изображения в Telegram: {e}")

# Функция для обработки изображений
async def process_image(img_url, bot, chat_id):
    try:
        # Загрузка изображения
        img_response = requests.get(img_url, timeout=10)
        img_response.raise_for_status()
        
        # Открытие изображения
        img = Image.open(BytesIO(img_response.content))
        
        # Изменение размера изображения до 640x640 с сохранением пропорций
        img.thumbnail((640, 640))
        
        # Создание нового изображения с белым фоном
        new_img = Image.new('RGB', (640, 640), (255, 255, 255))
        new_img.paste(img, ((640 - img.width) // 2, (640 - img.height) // 2))

        # Конвертация изображения в массив для OpenCV
        img_cv = np.array(new_img)

        # Объединённая функция для обработки результатов детекции
        def detect_and_draw(model, img_cv):
            results = model(img_cv)
            detected = False
            labels = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = box.conf[0].item()

                    if confidence >= 0.20:
                        detected = True
                        class_name = model.names[cls_id]

                        # Создаем подпись с названием класса и уверенностью
                        label = f"{class_name} {confidence:.2f}"
                        labels.append(label)

            return detected, labels

        # Запускаем детекцию для обеих моделей
        detected_rfr, labels_rfr = detect_and_draw(rfr, img_cv)
        detected_mollusc, labels_mollusc = detect_and_draw(mollusc, img_cv)

        # Объединяем метки от обеих моделей
        all_labels = labels_rfr + labels_mollusc

        if all_labels:
            # Объединяем все метки в одну строку
            label_text = "\n".join(all_labels)

            # Отправляем изображение в Telegram с подписью
            await send_image_to_telegram(new_img, chat_id, bot, label_text)

    except Exception as e:
        print(f'Ошибка при обработке изображения {img_url}: {e}')

# Основная функция для перебора солов и камер
async def main():
    sol = 0  # Начальный марсианский день
    cameras = ['FHAZ', 'RHAZ', 'MAST', 'CHEMCAM', 'MAHLI', 'MARDI', 'NAVCAM']  # Список камер

    # Инициализация Telegram-бота
    application = Application.builder().token(telegram_token).build()

    while True:
        for camera in cameras:
            # Параметры запроса
            params = {
                'sol': sol,
                'camera': camera,
                'api_key': api_key
            }
            
            try:
                # Отправка запроса к API
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                photos = data.get('photos', [])
                
                if photos:
                    for photo in photos:
                        img_url = photo['img_src']
                        await process_image(img_url, application.bot, chat_id)
                else:
                    print(f'Нет фотографий для сола {sol} и камеры {camera}.')
            except Exception as e:
                print(f'Ошибка при обращении к API для сола {sol} и камеры {camera}: {e}')

        time.sleep(50)  # Задержка перед следующим запросом
        sol += 1  # Переход к следующему марсианскому дню

if __name__ == '__main__':
    asyncio.run(main())
