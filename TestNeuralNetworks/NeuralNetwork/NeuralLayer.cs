using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestNeuralNetworks.NeuralNetwork
{
    /// <summary>
    /// An individual layer in the neural network
    /// </summary>
    class NeuralLayer {

        // Number of inputs we have
        private int inputNum;
        // Number of outputs we have
        private int outputNum;

        private Random rand = new Random(1);

        public float learningRate;
        public float[] inputs;
        public float[] outputs;
        public float[] gamma;
        public float[,] weights;
        public float[,] deltaWeights;

        public NeuralLayer(int inputNum, int outputNum, float learningRate) {
            this.inputNum = inputNum;
            this.outputNum = outputNum;
            this.learningRate = learningRate;
            outputs = new float[outputNum];
            inputs = new float[inputNum];
            weights = new float[outputNum, inputNum];
            deltaWeights = new float[outputNum, inputNum];
            gamma = new float[outputNum];

            InitWeights();
        }

        private void InitWeights() {
            for (int i = 0; i < outputNum; i++) {
                for (int j = 0; j < inputNum; j++) {
                    weights[i, j] = (float)rand.NextDouble();
                }
            }
        }

        private float TanHDerivative(float v) {
            return 1 - (v * v);
        }

        public float[] FeedForward(float[] inputs) {

            this.inputs = inputs;

            for (int i = 0; i < outputNum; i++) {
                outputs[i] = 0; 
                for (int j = 0; j < inputNum; j++)
                {
                    outputs[i] += inputs[j] * weights[i, j];
                }

                outputs[i] = (float)Math.Tanh(outputs[i]);
            }


            return outputs;
        }

        public void BackPropagation(float[] nextGamma, float[,] nextWeights)
        {
            for (int i = 0; i < outputNum; i++) {
                gamma[i] = 0;

                for (int j = 0; j < nextGamma.Length; j++) {
                    gamma[i] += nextGamma[j] * nextWeights[j, i];
                }

                gamma[i] *= TanHDerivative(outputs[i]);
            }

            for (int i = 0; i < outputNum; i++)
            {
                for (int j = 0; j < inputNum; j++)
                {
                    deltaWeights[i, j] = gamma[i] * inputs[j];
                }
            }
        }

        public void BackPropagationOutput(float[] expected)
        {
            for (int i = 0; i < outputNum; i++)
                gamma[i] = (outputs[i] - expected[i]) * TanHDerivative(outputs[i]);

            for (int i = 0; i < outputNum; i++) {
                for (int j = 0; j < inputNum; j++) {
                    deltaWeights[i, j] = gamma[i] * inputs[j];
                }
            }

           
        }

        public void UpdateWeights() {
            for (int i = 0; i < outputNum; i++)
            {
                for (int j = 0; j < inputNum; j++)
                    weights[i, j] -= deltaWeights[i, j] * learningRate;
            }
        }

    }
}
