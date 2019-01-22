using Layers;
using System;
using System.Collections.Generic;
using Functions;
using NetSerializer;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	class FeedForwardNet
	{
		private List<Layer> Layers = new List<Layer>();
		private List<int> Connections = new List<int>();
		private Random rnd;

		public FeedForwardNet()
		{
			rnd = new Random();
		}
        public List<Layer> GetLayers()
        {
            return Layers;
        }
        public void AddLayer(Layer layer)
        {
            if (Layers.Count > 0)
            {
                if (Layers[Layers.Count - 1].NeuronsCount != layer.InputVectorSize)
                    throw new Exception("Cant add layer");
                Layers[Layers.Count - 1].isOutput = false;
            }
            Layers.Add(layer);
        }
        public void AddLayer(int neurons, int inputs, FunctionType type)
        {
            if (Layers.Count > 0)
            {
                if (Layers[Layers.Count - 1].NeuronsCount != inputs)
                    throw new Exception("Cant add layer");
                Layers[Layers.Count - 1].isOutput = false;
            }
            Layers.Add(new Layer(inputs, neurons, true, rnd, type));
        }
        public void AddGaussianLayer(int neurons, int inputs)
        {
            AddLayer(neurons, inputs, FunctionType.Gaussian);
        }
        public void AddSigmoidLayer(int neurons, int inputs)
		{
            AddLayer(neurons, inputs, FunctionType.Sigmoid);
		}
		public void AddTangentialLayer(int neurons, int inputs)
		{
            AddLayer(neurons, inputs, FunctionType.Tangential);
		}
		public List<double> Calculate(List<double> input)
		{
			List<double> output = null;
			for(int i = 0; i < Layers.Count; ++i)
			{
				output = Layers[i].Calculate(i == 0 ? input : output);
			}
			return output;
		}

		private void Teach(List<double> input, List<double> output, double LearningRate)
		{
			List<List<double>> outputs = new List<List<double>>(Layers.Count);
			for(int i = 0; i < Layers.Count; ++i)
			{
				outputs.Add(Layers[i].Calculate(i == 0 ? input : outputs[i - 1]));
			}
			List<double> error = null;
			for(int i = Layers.Count - 1; i >= 0; --i)
			{
				error = Layers[i].Teach(
					i > 0 ? outputs[i - 1] : input,
					i == Layers.Count - 1 ? output : error,
					LearningRate);
			}
		}
        public void Collect(List<double> input, List<double> output, double LearningRate)
        {
            List<List<double>> outputs = new List<List<double>>(Layers.Count);
            for (int i = 0; i < Layers.Count; ++i)
            {
                outputs.Add(Layers[i].Calculate(i == 0 ? input : outputs[i - 1]));
            }
            List<double> error = null;
            for (int i = Layers.Count - 1; i >= 0; --i)
            {
                error = Layers[i].Collect(
                    i > 0 ? outputs[i - 1] : input,
                    i == Layers.Count - 1 ? output : error,
                    LearningRate);
            }
        }

        public void Train(List<List<double>> inputs, List<List<double>> outputs,
            double LearningRate, int epochs, int batchSize, bool writeNet = true)
        {
            DateTime startTime = DateTime.Now;
            DateTime prevTime = DateTime.Now;
            int procents = 0;
            Console.ForegroundColor = ConsoleColor.Green;

            for (int i = 0; i < epochs; ++i)
            {
                for (int j = 0; j < inputs.Count; ++j)
                {
                    Collect(inputs[j], outputs[j], LearningRate);
                    if (i != 0 && j != 0 && (i * j + j) % batchSize == 0)
                    {
                        ApplyBatch();
                    }
                }
                if (i / (epochs / 100) != procents)
                {
                    procents = i / (epochs / 100);
                    int tms = (int)((DateTime.Now - prevTime).TotalMinutes * (100 - procents));
                    if (tms != 0)
                    {
                        Console.Write("\rЗавершено на " + procents + "% Осталось " + tms + " минут SSE=" + SSE(inputs, outputs));
                    }
                    else
                    {
                        tms = (int)((DateTime.Now - prevTime).TotalSeconds * (100 - procents));
                        Console.Write("\rЗавершено на " + procents + "% Осталось " + tms + " секунд SSE=" + SSE(inputs, outputs));
                    }
                    prevTime = DateTime.Now;
                }
            }
            ApplyBatch();
            Console.Clear();
            Console.WriteLine("\rDone!");
            DateTime endTime = DateTime.Now;
            Console.WriteLine("Время обучения: " + (endTime - startTime));

            if(writeNet) ANNSerializer.WriteNet(this, "first_net", "1");
        }

        public void ApplyBatch()
        {
            for(int i = 0; i < Layers.Count; ++i)
            {
                Layers[i].ApplyDeltas();
            }
        }

        private double SSE(List<List<double>> inputs, List<List<double>> outputs)
        {
            double result = 0;
            for(int i = 0; i < inputs.Count; ++i)
            {
                List<double> calc = Calculate(inputs[i]);
                for (int j = 0; j < calc.Count; ++j)
                {
                    result += (calc[j] - outputs[i][j]) * (calc[j] - outputs[i][j]);
                }
            }
            return result;
        }

    }

    class LSTMCell
    {
        private Layer[] layers = new Layer[4];
        private int inputSize { get; set; }
        public LSTMCell(int input)
        {
            Random rnd = new Random();
            inputSize = input;
            layers[0] = new Layer(input * 2, input, false, rnd, FunctionType.Sigmoid);
            layers[1] = new Layer(input * 2, input, false, rnd, FunctionType.Sigmoid);
            layers[2] = new Layer(input * 2, input, false, rnd, FunctionType.Tangential);
            layers[3] = new Layer(input * 2, input, false, rnd, FunctionType.Sigmoid);
        }
        public List<Layer> GetLayers()
        {
            return new List<Layer>(layers);
        }
        private List<double> PointTh(List<double> first)
        {
            List<double> result = new List<double>(first.Count);
            for (int i = 0; i < first.Count; ++i)
            {
                result.Add(Math.Tanh(first[i]));
            }
            return result;
        }
        private List<double> PointThDer(List<double> tanh)
        {
            List<double> result = new List<double>(tanh.Count);
            for(int i = 0; i < tanh.Count; ++i)
            {
                result.Add(1.0 - tanh[i] * tanh[i]);
            }
            return result;
        }
        private List<double> PointSum(List<double> first, List<double> second)
        {
            if (first.Count != second.Count)
                throw new Exception("Cant use point operation");
            List<double> result = new List<double>(first.Count);
            for(int i = 0; i < first.Count; ++i)
            {
                result.Add(first[i] + second[i]);
            }
            return result;
        }
        private List<double> PointMul(List<double> first, List<double> second)
        {
            if (first.Count != second.Count)
                throw new Exception("Cant use point operation");
            List<double> result = new List<double>(first.Count);
            for (int i = 0; i < first.Count; ++i)
            {
                result.Add(first[i] * second[i]);
            }
            return result;
        }
        private List<double> PointMul(List<double> first, double second)
        {
            List<double> result = new List<double>(first.Count);
            for (int i = 0; i < first.Count; ++i)
            {
                result.Add(first[i] * second);
            }
            return result;
        }
        public List<double> Calculate(List<List<double>> input)
        {
           // List<List<double>> inp = new List<List<double>>(input);
            List<double> cellState = new List<double>(inputSize);
            for(int i = 0; i < inputSize; ++i)
            {
                cellState.Add(0);
            }
            List<double> result = new List<double>(inputSize);
            for(int i = 0; i < inputSize; ++i)
            {
                result.Add(0);
            }
            for(int i = 0; i < input.Count; ++i)
            {
                List<double> fullInput = new List<double>(input[i]);
                fullInput.AddRange(result);

                List<double> out0 = layers[0].Calculate(fullInput);
                List<double> out1 = layers[1].Calculate(fullInput);
                List<double> out2 = layers[2].Calculate(fullInput);
                List<double> out3 = layers[3].Calculate(fullInput);
                result = new List<double>(out0.Count);

                cellState = PointMul(cellState, out0);
                cellState = PointSum(PointMul(out1, out2), cellState);
                result = PointTh(cellState);
                result = PointMul(result, out3);
            }
            return result;
        }

        public void Teach(List<List<double>> input, List<double> output, double LearningRate)
        {
            List<List<double>> cellStates = new List<List<double>>();
            List<List<double>> cellStatesTanh = new List<List<double>>();
            List<List<double>> cellStatesTanhDer = new List<List<double>>();

            List<List<double>> outsF = new List<List<double>>(input.Count);
            List<List<double>> outsI = new List<List<double>>(input.Count);
            List<List<double>> outsC = new List<List<double>>(input.Count);
            List<List<double>> outsO = new List<List<double>>(input.Count);

            cellStates.Add(new List<double>(inputSize));
            cellStatesTanh.Add(new List<double>(inputSize));
            cellStatesTanhDer.Add(new List<double>(inputSize));
            for (int i = 0; i < inputSize; ++i)
            {
                cellStates[0].Add(0);
                cellStatesTanh[0].Add(0);
                cellStatesTanhDer[0].Add(0);
            }
            List<List<double> > result = new List<List<double>>();
            result.Add(new List<double>(inputSize));
            for (int i = 0; i < inputSize; ++i)
            {
                result[0].Add(0);
            }
            List<List<double>> fullinputs = new List<List<double>>();
            for (int i = 0; i < input.Count; ++i)
            {
                if (input[i].Count != inputSize)
                    throw new Exception("input vector in not valid");

                List<double> fullInput = new List<double>(input[i]);
                fullInput.AddRange(result[i]);
                fullinputs.Add(fullInput);

                outsF.Add(layers[0].Calculate(fullInput));
                outsI.Add(layers[1].Calculate(fullInput));
                outsC.Add(layers[2].Calculate(fullInput));
                outsO.Add(layers[3].Calculate(fullInput));

                cellStates.Add(PointSum(PointMul(outsI[i], outsC[i]),
                    PointMul(cellStates[i], outsF[i])));
                cellStatesTanh.Add(PointTh(cellStates[i + 1]));
                cellStatesTanhDer.Add(PointThDer(cellStatesTanh[i + 1]));
                result.Add(PointMul(cellStatesTanh[i + 1], outsO[i]));
            }

            List<double> error = new List<double>(inputSize);
            for(int i = 0; i < inputSize; ++i)
            {
                error.Add(output[i] - result[result.Count - 1][i]);
            }
            List<double> errorF = error;
            List<double> errorI = error;
            List<double> errorC = error;
            List<double> errorO = error;
            
            for(int i = input.Count - 1; i >= 0; --i)
            {
                List<double> tmp = PointMul(PointMul(outsO[i], cellStatesTanhDer[i + 1]), LearningRate);
                List<double> deltaF = PointMul(cellStates[i], tmp);
                List<double> deltaI = PointMul(PointMul(outsC[i], tmp), 0.1);
                List<double> deltaC = PointMul(outsI[i], tmp);
                List<double> deltaO = PointMul(cellStatesTanh[i + 1], LearningRate);

                errorF = layers[0].Collect4LSTM(fullinputs[i], errorF, /*LearningRate*/deltaF);
                errorI = layers[1].Collect4LSTM(fullinputs[i], errorI, /*LearningRate*/deltaI);
                errorC = layers[2].Collect4LSTM(fullinputs[i], errorC, /*LearningRate*/deltaC);
                errorO = layers[3].Collect4LSTM(fullinputs[i], errorO, /*LearningRate*/deltaO);
            }

            for(int i = 0; i < 4; ++i)
            {
                layers[i].ApplyDeltas();
            }
        }
    }
}
