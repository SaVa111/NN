# NN
Простанство имен .. представляет собой библиотеку для работы с искусственными нейронными сетями.

Класс FeedForwardNet является нейронной сетью прямого распространения 
Класс поддерживает слои нейронов с Логистической, тангенсальной и гаусовой функцией активации
Для этого используется функция AddLayers, где указывается соответсвующая функция активации, либо соответсвующие функции.
```csharp
net = new FeedForwardNet();
net.AddGaussianLayer(10, 4);
net.AddSigmoidLayer(5, 10);
net.AddTangentialLayer(3, 5);
```
Обучение осуществляет функция Train
```csharp
net.Train(inputs, outputs, LearningRate, epochs, batchSize);
```
inputs -- List<List<double> > Список входов обучающей выборки.
outputs -- List<List<double> > Список выходов обучающей выборки.
LearningRate -- double Коэфициент скорости обучения сети (По умолчанию 0.01).
epochs -- int Колличество эпох.
batchSize -- int размер пакета (По умолчанию 1).
