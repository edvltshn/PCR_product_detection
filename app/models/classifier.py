import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from efficientnet_pytorch import EfficientNet


model = EfficientNet.from_name('efficientnet-b7')  # Создаем модель
num_ftrs = model._fc.in_features  # Получаем количество входных признаков в последний слой
model._fc = torch.nn.Linear(num_ftrs, 3)  # Заменяем последний слой на новый, имеющий нужное количество выходов
model.load_state_dict(torch.load('models/EffNetB7_finetuned.pth'))  # Загружаем веса модели
model.eval()  # Переводим модель в режим оценки


# Определение трансформации
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1] for each channel
])


# Классы изображений
classes = ['no', 'r', 'yes'] 


def classify_image(image_file):
    # Используйте .file, чтобы получить SpooledTemporaryFile
    temp_file = image_file.file
    # Переведите указатель в начало файла, если он уже был прочитан
    temp_file.seek(0)
    # Создайте BytesIO объект из содержимого SpooledTemporaryFile
    image_bytes_io = BytesIO(temp_file.read())
    # Откройте изображение с помощью PIL
    image = Image.open(image_bytes_io)


    tensor = transform(image).unsqueeze(0)
        

    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0].tolist()



    result = {
        'predicted_class': classes[predicted.item()],
        'prediction_probabilities': {classes[j]: probabilities[j] for j in range(len(classes))},
    }


    return result
