import requests
from PIL import Image
from io import BytesIO
import os
from ultralytics import YOLO

# Ваш API-ключ NASA
api_key = ''

# Базовый URL для API Mars Rover Photos
api_url = 'https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos'

# Инициализация модели YOLOv8
model = YOLO('yolov8n.pt')  # Используем предобученную модель YOLOv8n

# Функция для обработки изображений
def process_image(img_url):
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
        
        # Выполнение детекции объектов
        results = model(new_img)
        
        # Обработка результатов детекции
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = box.conf[0].item()
                if confidence >= 0.45:
                    # Получение имени класса
                    class_name = model.names[cls_id]
                    
                    # Создание директории для класса, если не существует
                    if not os.path.exists(class_name):
                        os.makedirs(class_name)
                    
                    # Сохранение изображения в соответствующую папку
                    img_name = os.path.basename(img_url)
                    save_path = os.path.join(class_name, img_name)
                    new_img.save(save_path)
                    
                    print(f'Изображение сохранено как {save_path} с классом {class_name} и уверенностью {confidence:.2f}')
    except Exception as e:
        print(f'Ошибка при обработке изображения {img_url}: {e}')

# Основная функция для перебора солов и камер
def main():
    sol = 0  # Начальный марсианский день
    cameras = ['FHAZ', 'RHAZ', 'MAST', 'CHEMCAM', 'MAHLI', 'MARDI', 'NAVCAM']  # Список камер

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
                        process_image(img_url)
                else:
                    print(f'Нет фотографий для сола {sol} и камеры {camera}.')
            except Exception as e:
                print(f'Ошибка при обращении к API для сола {sol} и камеры {camera}: {e}')
        
        sol += 1  # Переход к следующему марсианскому дню

if __name__ == '__main__':
    main()
