//#define PARALLEL

using System;
using System.Text;
using System.Collections.Generic;
using NeuralNetwork;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using NetSerializer;

namespace ConsoleApplication1
{
	public static class Program
    {
		static List<List<double>> ReadNetIO(string filepath)
		{
			List<List<double>> result = new List<List<double>>();
			using(StreamReader sr = new StreamReader(filepath))
			{
				string s;
				while((s = sr.ReadLine()) != null)
				{
					List<double> ls = new List<double>();
					string[] strings = s.Split(new char[] {' '});
					for(int i = 0; i < strings.Length; ++i)
					{
						ls.Add(Convert.ToDouble(strings[i]));
					}
					result.Add(ls);
				}
			}
			return result;
		}
        static string ConvertToIris(List<double> vec)
        {
            if (vec.Count != 3)
                throw new Exception("Cannot convert to iris");

            if (vec[0] >= vec[1] && vec[0] >= vec[2])
                return "setosa";
            if (vec[1] >= vec[0] && vec[1] >= vec[2])
                return "verticolor";
            if (vec[2] >= vec[0] && vec[2] >= vec[1])
                return "verginia";
            return "<unknown>";
        }

        public static void Shuffle<T>(this IList<T> list)
        {
            Random rng = new Random(12);
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        static void Main(string[] args)
        {
			List<List<double>> inputs = ReadNetIO(@"input4.txt");
			List<List<double>> outputs = ReadNetIO(@"output3.txt");

            Shuffle(inputs);
            Shuffle(outputs);

            List<List<double>> testin = new List<List<double>>();
            List<List<double>> testout = new List<List<double>>();

            for(int i = 1; i < 11; ++i)
            {
                testin.Add(inputs[inputs.Count - i]);
                testout.Add(outputs[outputs.Count - i]);
                inputs.RemoveAt(inputs.Count - i);
                outputs.RemoveAt(outputs.Count - i);
            }

            const bool newNet = true;
            FeedForwardNet net;

            if (newNet)
            {
                const double LearningRate = 0.1;
                const int epochs = 10000;
                const int batchSize = 1;
                const int type = 1;
                net = new FeedForwardNet();
                if (type == 1)
                {
                    net.AddSigmoidLayer(10, 4);
                    net.AddSigmoidLayer(5, 10);
                    net.AddSigmoidLayer(3, 5);
                }
                else if (type == 2)
                {
                    net.AddTangentialLayer(10, 4);
                    net.AddTangentialLayer(5, 10);
                    net.AddTangentialLayer(3, 5);
                }
                else if (type == 3)
                {
                    net.AddGaussianLayer(10, 4);
                    net.AddGaussianLayer(5, 10);
                    net.AddGaussianLayer(3, 5);
                }
                else if (type == 4)
                {
                    net.AddGaussianLayer(10, 4);
                    net.AddSigmoidLayer(5, 10);
                    net.AddTangentialLayer(3, 5);
                }

                net.Train(inputs, outputs, LearningRate, epochs, batchSize);
            }
            else
            {
                net = ANNSerializer.ReadNet(@"first_net/1.ann");
            }
            
			for(int i = 0; i < outputs.Count; ++i)
			{
				List<double> result = net.Calculate(inputs[i]);
                Console.ForegroundColor = ConsoleColor.White;
                for (int j = 0; j < inputs[i].Count; ++j)
                {
                    Console.Write(inputs[i][j] + " ");
                }
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Yellow;
				Console.Write(i + " " + ConvertToIris(result) + " ");
                Console.ForegroundColor = ConsoleColor.Green;
				Console.WriteLine(ConvertToIris(outputs[i]) + "\n");
			}
            Console.WriteLine("Tests:");
            for (int i = 0; i < testin.Count; ++i)
            {
                List<double> result = net.Calculate(testin[i]);
                Console.ForegroundColor = ConsoleColor.White;
                for (int j = 0; j < testin[i].Count; ++j)
                {
                    Console.Write(testin[i][j] + " ");
                }
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write(i + " " + ConvertToIris(result) + " ");
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine(ConvertToIris(testout[i]) + "\n");
            }
        }
	}
}
