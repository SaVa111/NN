# NN

```csharp
net = new FeedForwardNet();
net.AddGaussianLayer(10, 4);
net.AddSigmoidLayer(5, 10);
net.AddTangentialLayer(3, 5);
net.Train(inputs, outputs, LearningRate, epochs, batchSize);
```
