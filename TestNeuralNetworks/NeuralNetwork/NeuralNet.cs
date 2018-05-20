using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestNeuralNetworks.NeuralNetwork
{
    class NeuralNet {

        NeuralLayer[] layers;
        public float learningRate;

        public NeuralNet(int[] topology, float lRate) {

            layers = new NeuralLayer[topology.Length - 1];
            learningRate = lRate;

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new NeuralLayer(topology[i], topology[i + 1], learningRate);
            }

        }

        public float[] FeedForward(float[] inputs) {

            layers[0].FeedForward(inputs);
            for (int i = 1; i < layers.Length; i++)
                layers[i].FeedForward(layers[i - 1].outputs);

            return layers[layers.Length - 1].outputs;
        }

        public void BackPropagation(float[] expected) {

            for (int i = layers.Length-1; i >= 0; i--) {
                if (i == layers.Length - 1)
                    layers[i].BackPropagationOutput(expected);
                else
                    layers[i].BackPropagation(layers[i+1].gamma, layers[i+1].weights);
            }


            for (int i = 0; i < layers.Length; i++) {
                layers[i].UpdateWeights();
            }


        }

    }
}
