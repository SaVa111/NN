using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;
using Layers;
using System.IO;

namespace NetSerializer
{
	static class ANNSerializer
	{
        /**************************************************
         * Construction of .ann file
         * (4)int32 count of layers
         * for each layer:
         *** (1)FunctionType type of layer
         *** (4)int32 count of neurons
         *** for each neuron:
         ***** (4)int32 size of input vector
         ***** (8 * size)double[] array of weights
         **************************************************/

        static public FeedForwardNet ReadNet(string filepath)
		{
            int position = 0;
            FeedForwardNet result = new FeedForwardNet();
            byte[] bytes = File.ReadAllBytes(filepath);
            int layersCount = BitConverter.ToInt32(bytes, position);
            position += 4;
            for(int i = 0; i < layersCount; ++i)
            {
                Functions.FunctionType type =
                    (Functions.FunctionType)bytes[position];
                ++position;
                int neuronsCount = BitConverter.ToInt32(bytes, position);
                position += 4;
                List<List<double>> toLayer = new List<List<double>>(neuronsCount);
                for(int j = 0; j < neuronsCount; ++j)
                {
                    int inputVectorSize = BitConverter.ToInt32(bytes, position);
                    List<double> weights = new List<double>(inputVectorSize);
                    position += 4;
                    for(int l = 0; l < inputVectorSize; ++l)
                    {
                        weights.Add(BitConverter.ToDouble(bytes, position));
                        position += 8;
                    }
                    toLayer.Add(weights);
                }
                result.AddLayer(new Layer(toLayer, true, type));
            }
            return result;
		}
		static public void WriteNet(FeedForwardNet net, string netname, string filename)
		{
            Directory.CreateDirectory(netname);
            string filepath = netname + @"/" + filename + ".ann";
            List<byte> toWrite = new List<byte>();
            List<Layer> Layers = net.GetLayers();
            toWrite.AddRange(BitConverter.GetBytes(Layers.Count));
            for(int i = 0; i < Layers.Count; ++i)
            {
                List<List<double>> weights = Layers[i].GetWeghts();
                toWrite.Add((byte)Layers[i].GetLayerType());                
                toWrite.AddRange(BitConverter.GetBytes(weights.Count));

                for(int j = 0; j < weights.Count; ++j)
                {
                    toWrite.AddRange(BitConverter.GetBytes(weights[j].Count));
                    for(int l = 0; l < weights[j].Count; ++l)
                    {
                        toWrite.AddRange(BitConverter.GetBytes(weights[j][l]));
                    }
                }
            }
            File.WriteAllBytes(filepath, toWrite.ToArray());
		}
        static public void WriteNet(LSTMCell net, string netname, string filename)
        {
            Directory.CreateDirectory(netname);
            string filepath = netname + @"/" + filename + ".ann";
            List<byte> toWrite = new List<byte>();
            List<Layer> Layers = net.GetLayers();
            toWrite.AddRange(BitConverter.GetBytes(Layers.Count));
            for (int i = 0; i < Layers.Count; ++i)
            {
                List<List<double>> weights = Layers[i].GetWeghts();
                toWrite.Add((byte)Layers[i].GetLayerType());
                toWrite.AddRange(BitConverter.GetBytes(weights.Count));

                for (int j = 0; j < weights.Count; ++j)
                {
                    toWrite.AddRange(BitConverter.GetBytes(weights[j].Count));
                    for (int l = 0; l < weights[j].Count; ++l)
                    {
                        toWrite.AddRange(BitConverter.GetBytes(weights[j][l]));
                    }
                }
            }
            File.WriteAllBytes(filepath, toWrite.ToArray());
        }
    }
}
