using System;
using System.Collections.Generic;
using Functions;

namespace Neurons
{
	class Neuron : ICloneable
	{
        private List<double> weight;
        private List<double> deltas;
        private ActivationFunction func;
	    public Neuron() {}

        public object Clone()
        {
            return new Neuron(weight, func.GetFunctionType());
        }

		public Neuron(List<double> toSet, FunctionType type)
		{
            func = ActivationFunction.GetFunction(type);
            weight = new List<double>(toSet);
            deltas = new List<double>(weight.Count);
            for(int i = 0; i < weight.Count; ++i)
            {
                deltas.Add(0);
            }
		}
        public Neuron(int size, FunctionType type, Random rnd)
        {
            weight = new List<double>(size + 1);
            func = ActivationFunction.GetFunction(type);
            for (int i = 0; i < size + 1; ++i)
            {
                int value = rnd.Next(20000) - 10000;
                weight.Add(value / 10000.0);
            }
            deltas = new List<double>(weight.Count);
            for (int i = 0; i < weight.Count; ++i)
            {
                deltas.Add(0);
            }
        }
        public FunctionType GetFunctionType()
        {
            return func.GetFunctionType();
        }
        public List<double> GetWeight()
		{
			return new List<double>(weight);
		}
        public double Calculate(List<double> input)
        {
            input.Add(1);
            if (input.Count != weight.Count)
                throw new Exception("Input vector for neuron is not valid");
            
            double Sum = 0;
            for (int i = 0; i < weight.Count; ++i)
            {
                Sum += weight[i] * input[i];
            }
            input.RemoveAt(input.Count - 1);
            FunctionDel f = func.GetFunction();
            return f(Sum);
        }
        public List<double> Teach(
            List<double> input, double output, double LearningRate, bool isOutput)
        {
            input.Add(1);
            if (input.Count != weight.Count)
                throw new Exception("Input vector for neuron is not valid");

            double Sum = 0;
            for (int i = 0; i < weight.Count; ++i)
            {
                Sum += weight[i] * input[i];
            }
            FunctionDel f = func.GetFunction();
            FunctionDel fDer = func.GetDerivative();
            double error = (isOutput ? (f(Sum) - output) : output);
            double delta = error * fDer(Sum);
            for (int i = 0; i < weight.Count; ++i)
            {
                weight[i] -= input[i] * delta * LearningRate;
            }
            List<double> ret = new List<double>();
            input.RemoveAt(input.Count - 1);
            for (int i = 0; i < weight.Count - 1; ++i)
            {
                ret.Add(delta * weight[i]);
            }
            return ret;
        }
        public void ApplyDeltas()
        {
            for(int i = 0; i < deltas.Count; ++i)
            {
                weight[i] += deltas[i];
            }
            deltas.Clear();
            for(int i = 0; i < weight.Count; ++i)
            {
                deltas.Add(0);
            }
        }
        public List<double> CollectDeltas(
            List<double> input, double output, double LearningRate, bool isOutput)
        {
            input.Add(1);
            if (input.Count != weight.Count)
                throw new Exception("Input vector for neuron is not valid");

            double Sum = 0;
            for (int i = 0; i < weight.Count; ++i)
            {
                Sum += weight[i] * input[i];
            }
            FunctionDel f = func.GetFunction();
            FunctionDel fDer = func.GetDerivative();
            double error = (isOutput ? (f(Sum) - output) : output);
            double delta = error * fDer(Sum) * LearningRate;
            for (int i = 0; i < deltas.Count; ++i)
            {
                deltas[i] -= input[i] * delta;
            }
            List<double> ret = new List<double>();
            input.RemoveAt(input.Count - 1);
            for (int i = 0; i < weight.Count - 1; ++i)
            {
                ret.Add(delta * (weight[i] + deltas[i]));
            }
            return ret;
        }
    }
}