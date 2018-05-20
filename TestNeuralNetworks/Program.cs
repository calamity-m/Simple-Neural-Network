using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TestNeuralNetworks.NeuralNetwork;

namespace TestNeuralNetworks
{
    class Program
    {
        static void Main(string[] args)
        {

            Console.WriteLine("Hello");
            NeuralNet neuralNetwork = new NeuralNet(new int[] { 3, 35, 1 }, 0.0333f, 1345);
            int epochs = 5000;

            for (int i = 0; i < epochs; i++) {
                neuralNetwork.FeedForward(new float[] { 0, 0, 0 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 0, 0, 1 });
                neuralNetwork.BackPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 0, 1, 0 });
                neuralNetwork.BackPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 1, 0, 0 });
                neuralNetwork.BackPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 1, 1, 0 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 0, 1, 1 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 0, 1 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 1, 1 });
                neuralNetwork.BackPropagation(new float[] { 1 });
            }


            float[] result = neuralNetwork.FeedForward(new float[] { 0, 0, 0 });
            PrintFloatArray(result);
            result = neuralNetwork.FeedForward(new float[] { 1, 0, 0 });
            PrintFloatArray(result);
            result = neuralNetwork.FeedForward(new float[] { 1, 1, 0 });
            PrintFloatArray(result);
            result = neuralNetwork.FeedForward(new float[] { 0, 0, 0 });
            PrintFloatArray(result);
            result = neuralNetwork.FeedForward(new float[] { 1, 1, 1 });
            PrintFloatArray(result);
            result = neuralNetwork.FeedForward(new float[] { 1, 1, 1 });
            PrintFloatArray(result);
            result = neuralNetwork.FeedForward(new float[] { 0, 1, 0 });
            PrintFloatArray(result);

            Console.WriteLine("Finished");
        }

        public static void PrintFloatArray(float[] arr)
        {
            for (int i = 0; i < arr.Length; i++)
                Console.Write(arr[i] + " ");

            Console.WriteLine("");
        }
    }
}
