# Сверточная нейронная сеть для классификации изображений

## Описание проекта

Этот проект направлен на создание и обучение сверточной нейронной сети (CNN) для классификации изображений размером 48x48x1 в один из 10 классов.


## Структура данных

- **Формат изображений**: 48x48 пикселей, 1 канал (градации серого).
- **Целевая переменная**: Целое число от 0 до 9, представляющее класс изображения.
- **Количество классов**: 10.

## Архитектура модели

Сверточная нейронная сеть построена с использованием PyTorch и состоит из следующих слоев:

- **Вход:** Изображения 48x48x1.
- **Первый сверточный блок**:
  - Сверточный слой: Conv2d, 6 каналов, ядро 3x3.
  - Активация: ReLU.
  - Пулинг: AvgPool2d, ядро 2x2.
- **Второй сверточный блок**:
  - Сверточный слой: Conv2d, 3 канала, ядро 3x3.
  - Активация: ReLU.
  - Пулинг: MaxPool2d, ядро 2x2.
- **Растягивание**:
  - Преобразует карты активации в одномерный вектор для подачи полносвязной нейронной сети.
- **Полносвязные слои (Fully Connected Layers)**:
  - Слой 1: 300 входов → 120 выходов, с активацией `ReLU`.
  - Слой 2: 120 входов → 84 выхода, с активацией `Sigmoid`.
  - Выходной слой: 84 входа → 10 выходов (для предсказания класса).
```python
#input 48x48x1
def create_cnn():
    model = nn.Sequential(
        # Первый сверточный слой: Conv2d, ReLU, AvgPool2d
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),  # out: 46x46x6
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),  # out: 24x24x6
        nn.ReLU(),

        # Второй сверточный слой: Conv2d, ReLU
        nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3),  # out: 21x21x3
        nn.ReLU(),

        # Второй слой пулинга: MaxPool2d
        nn.MaxPool2d(kernel_size=2, stride=2),  # out: 10x10x3

        # Растягиваем матрицу
        nn.Flatten(),

        # Подаем преобразованное изображение на вход FCL
        nn.Linear(10*10*3, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    return model
```
## Обучение модели

Модель обучается с использованием стохастического градиентного спуска `SGD` и функции потерь `CrossEntropyLoss`. В процессе обучения отслеживается точность на обучающем наборе данных для каждой эпохи.

### Код функции обучения
```python
def train_model(model, dataloader, device=device, num_epoch=NUM_EPOCH, learning_rate=LEARN_RATE):
  model = model.to(device)

  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  acc_history = []


  for epoch in range(num_epoch):
          correct = 0
          total = 0

          for i, batch in enumerate(dataloader):
              images, labels = batch
              images = images.to(device)
              labels = labels.to(device)

              optimizer.zero_grad()

              # Считаем лосс
              predicted_labels = model(images)
              loss = loss_function(predicted_labels, labels)

              loss.backward()

              # Обновление параметров
              optimizer.step()

              # Подсчет точности
              _, predicted = predicted_labels.max(1) # По столбцам выбираем индекс с максимальной верояностью
              total += labels.size(0)
              correct += predicted.eq(labels).sum().item()

          # Точность по эпохе
          acc = correct / total
          acc_history.append(acc)

  return model, acc_history
```
## Оценка модели

После обучения модели оценивается точность на обучающей и тестовой выборках. Это позволяет проверить, насколько хорошо модель обучилась на данных и насколько она обобщается на новые данные.
### Код для оценки точности
```python
def evaluate_model(model, dataloader, device=device):
    model = model.to(device)
    model.eval()  # Устанавливаем модель в режим оценки

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            predicted_labels = model(images)

            _, predicted = predicted_labels.max(1)  # Выбираем класс с максимальной вероятностью

            # Подсчитываем количество правильных предсказаний
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total  # Вычисляем точность

    return accuracy
```
