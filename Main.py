import requests
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO
import time
import os
import cv2

# Ваш API-ключ NASA
api_key = ''

# Ваш токен Telegram-бота
telegram_token = ''

# ID вашего чата (личка, либо группа)
chat_id = ''

# Базовый URL для API Mars Rover Photos
api_url = 'https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos'

# Инициализация моделей YOLOv8
rfr = YOLO('rfr.pt')  # Модель для поиска окаменелостей
mollusc = YOLO('mollusc.pt')  # Модель для поиска моллюсков (пока закомментирована)

# Путь к файлу для хранения последнего sol
sol_file = 'last_sol.txt'

# Функция для отправки изображений в Telegram
def send_image_to_telegram(img, chat_id, bot_token, label):
    try:
        # Преобразуем изображение в формат для отправки через Telegram
        bio = BytesIO()
        bio.name = 'image.jpg'  # Имя файла для Telegram
        img.save(bio, 'JPEG')
        bio.seek(0)

        # Составляем URL для отправки запроса через Telegram Bot API
        url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'

        # Формируем параметры для запроса
        files = {'photo': bio}
        data = {'chat_id': chat_id, 'caption': label}

        # Отправляем запрос
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            print("Изображение отправлено в Telegram с подписью.")
        else:
            print(f"Ошибка отправки изображения: {response.text}")
    except Exception as e:
        print(f"Ошибка отправки изображения в Telegram: {e}")

# Функция для обработки изображений
def process_image(img_url, bot_token, chat_id):
    try:
        # Загрузка изображения
        img_response = requests.get(img_url, timeout=10)
        img_response.raise_for_status()
        
        # Открытие изображения
        img = Image.open(BytesIO(img_response.content))
        
        # Изменение размера изображения до 640x640 с сохранением пропорций
        img = img.resize((640, 640))
        
        # Конвертация изображения в массив для OpenCV
        img_cv = np.array(img)
        
        # Проверяем, есть ли у изображения 3 канала (RGB)
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:  # Если RGBA
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
        elif len(img_cv.shape) == 2:  # Если изображение в градациях серого (без каналов)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)  # Преобразуем в RGB
        
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
            send_image_to_telegram(img, chat_id, bot_token, label_text)

    except Exception as e:
        print(f'Ошибка при обработке изображения {img_url}: {e}')

# Основная функция для перебора солов и камер
def main():
    sol = 0  # Начальный марсианский день
    cameras = ['FHAZ', 'RHAZ', 'MAST', 'CHEMCAM', 'MAHLI', 'MARDI', 'NAVCAM']  # Список камер

    # Ваш токен Telegram-бота
    bot_token = telegram_token

    # Чтение последнего sol, если файл существует
    if os.path.exists(sol_file):
        with open(sol_file, 'r') as f:
            sol = int(f.read().strip())
    print(f"Начинаю обработку с марсианского дня {sol}")

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
                        process_image(img_url, bot_token, chat_id)
                else:
                    print(f'Нет фотографий для сола {sol} и камеры {camera}.')
            except Exception as e:
                print(f'Ошибка при обращении к API для сола {sol} и камеры {camera}: {e}')

        # Сохранение последнего sol
        with open(sol_file, 'w') as f:
            f.write(str(sol))

        print(f"Задержка перед следующим циклом обработки для сола {sol}")
        time.sleep(50)  # Задержка перед следующим запросом
        sol += 1  # Переход к следующему марсианскому дню

if __name__ == '__main__':
    main()
