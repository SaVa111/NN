using Neurons;
using System;
using System.Collections.Generic;
using Functions;

namespace Layers
{
    class Layer : ICloneable
	{
        public bool isOutput { get; set; }
        private int inputSize;
        private List<Neuron> Neurons;
        private FunctionType LayerType;

        public object Clone()
        {
            return new Layer(GetWeghts(), isOutput, LayerType);
        }

        public Layer(
            int InputSize, int OutputSize,
            bool IsOutputLayer, Random rnd,
            FunctionType type)
		{
			isOutput = IsOutputLayer;
			inputSize = InputSize;
            LayerType = type;
			Neurons = new List<Neuron>();
            for (int i = 0; i < OutputSize; ++i)
            {
                Neurons.Add(new Neuron(InputSize, type, rnd));
            }
        }
        public Layer(
            List<List<double>> toSet,
            bool IsOutputLayer,
            FunctionType type)
        {
            isOutput = IsOutputLayer;
            Neurons = new List<Neuron>();
            LayerType = type;
            for(int i = 0; i < toSet.Count; ++i)
            {
                Neurons.Add(new Neuron(toSet[i], type));
            }
        }
        public int NeuronsCount
        {
            get { return Neurons.Count; }
        }
        public int InputVectorSize
        {
            get { return Neurons[0].GetWeight().Count - 1; }
        }

        public List<List<double>> GetWeghts()
		{
			List<List<double>> ret = new List<List<double>>();
			foreach(Neuron neuron in Neurons)
			{
				ret.Add(neuron.GetWeight());
			}
			return ret;
		}
		public List<double> Calculate(List<double> input)
		{
			List<double> res = new List<double>();
			foreach(Neuron neuron in Neurons)
			{
				res.Add(neuron.Calculate(input));
			}
			return res;
		}
        public FunctionType GetLayerType()
        {
            return LayerType;
        }

        public void ApplyDeltas()
        {
            for(int i = 0; i < Neurons.Count; ++i)
            {
                Neurons[i].ApplyDeltas();
            }
        }

		public List<double> Teach(
			List<double> input, List<double> errors, double LearningRate)
		{
			List<double> res = new List<double>(input.Count);
			for(int i = 0; i < input.Count; ++i)
			{
				res.Add(0);
			}
			for(int i = 0; i < Neurons.Count; ++i)
			{
				List<double> asd = Neurons[i].Teach(input, errors[i], LearningRate, isOutput);
				for(int j = 0; j < asd.Count; ++j)
				{
					res[j] += asd[j];
				}
			}
			return res;
		}

        public List<double> Collect(
            List<double> input, List<double> errors, double LearningRate)
        {
            List<double> res = new List<double>(input.Count);
            for (int i = 0; i < input.Count; ++i)
            {
                res.Add(0);
            }
            for (int i = 0; i < Neurons.Count; ++i)
            {
                List<double> asd = Neurons[i].CollectDeltas(input, errors[i], LearningRate, isOutput);
                for (int j = 0; j < asd.Count; ++j)
                {
                    res[j] += asd[j];
                }
            }
            return res;
        }

        public List<double> Collect4LSTM(
            List<double> input, List<double> errors, List<double> LearningRate)
        {
            List<double> res = new List<double>(input.Count);
            for (int i = 0; i < input.Count; ++i)
            {
                res.Add(0);
            }
            for (int i = 0; i < Neurons.Count; ++i)
            {
                List<double> asd = Neurons[i].CollectDeltas(input, errors[i], LearningRate[i], isOutput);
                for (int j = 0; j < asd.Count; ++j)
                {
                    res[j] += asd[j];
                }
            }
            return res;
        }
        public List<double> Collect4LSTM(
    List<double> input, List<double> errors, double LearningRate)
        {
            List<double> res = new List<double>(input.Count);
            for (int i = 0; i < input.Count; ++i)
            {
                res.Add(0);
            }
            for (int i = 0; i < Neurons.Count; ++i)
            {
                List<double> asd = Neurons[i].CollectDeltas(input, errors[i], LearningRate, isOutput);
                for (int j = 0; j < asd.Count; ++j)
                {
                    res[j] += asd[j];
                }
            }
            return res;
        }
    }
}