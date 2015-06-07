/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Besseym
 */
public class Network 
{
   List<Layer> layers = new ArrayList<>();

   Double bias = 1.0;
   
    public Network(int inputCount, List<Integer> numberOfNeuronPerLayer) 
    {
        
        if (numberOfNeuronPerLayer.isEmpty()) 
        {
            throw new UnsupportedOperationException("numberOfNeuronPerLayer is empty");
        }

        //add 1 for bias
        layers.add(new Layer(numberOfNeuronPerLayer.get(0), inputCount + 1));

        for (int i = 1; i < numberOfNeuronPerLayer.size(); i++) 
        {     
            layers.add(new Layer(numberOfNeuronPerLayer.get(i),
                    numberOfNeuronPerLayer.get(i - 1) + 1));
        }
    }

    public List<Double> getOutputs(List<Double> inputs)
    {
        
        List<Double> outputs = new ArrayList<>(inputs);

        for (Layer layer : layers) 
        { 
            addBias(outputs);
            
            outputs = layer.produceOutputs(outputs);
        }

        return outputs;
    } 
    
    public double train(List<Double> inputs, double classification, double learningValue) 
    {
        ArrayList<List<Double>> all = new ArrayList<>();
        List<Double> outputs = new ArrayList<>(inputs);
        
        // feed forward to calculate outputs
        for (Layer layer : layers) 
        {    
            outputs.add(bias);
            outputs = layer.produceOutputs(outputs);
            all.add(outputs);
            
        }

        ArrayList<ArrayList<Double>> allErrors = new ArrayList<>();
        // work backwards to calculate errors

        // do output nodes
        ArrayList<Double> error = new ArrayList<>();
        List<Double> currentOutputs = all.get(all.size() - 1);
        Layer current = layers.get(layers.size() - 1);
        
        for (int i = 0; i < current.neurons.size(); i++) 
        {
            double expected = (classification == i ? 1 : 0);
            error.add(currentOutputs.get(i) * (1 - currentOutputs.get(i)) * (currentOutputs.get(i) - expected));
        }

        allErrors.add(error);

        // hidden nodes are a different equation
        for (int i = layers.size() - 2; i >= 0; i--) 
        {
            // for each hidden layer
            current = layers.get(i);
            error = new ArrayList<>();
            outputs = all.get(i);
            ArrayList<Double> followingError = allErrors.get(0);
            
            for (int j = 0; j < current.neurons.size(); j++)
            {
                
                double sumError = 0;
                
                Layer nextLayer = layers.get(i + 1);
                for (int k = 0; k < followingError.size(); k++) 
                {
                    // for each neuron in following layer
                    sumError += followingError.get(k) * nextLayer.neurons.get(k).weights.get(j);
                }

                double errorVal = outputs.get(j) * (1 - outputs.get(j)) * sumError;
                error.add(errorVal);
            }

            allErrors.add(0, error);
        }

        // feed forward to update weights based on errors
        inputs.add(bias);
        all.add(0,inputs);
        for (int i = 0; i < layers.size(); i++) 
        {
            // foreach layer
            current = layers.get(i);
            for (int j = 0; j < current.neurons.size(); j++)
            {
                // foreach neuron in layer
                Neuron neuron = current.neurons.get(j);
                for (int k = 0; k < neuron.weights.size(); k++)
                {
                    double newWeight = neuron.weights.get(k) - all.get(i).get(k) * allErrors.get(i).get(j) * learningValue;
                    neuron.weights.set(k, newWeight);
                }
            }
        }

        // return total error
        double totalError = 0;
        for (List<Double> l : allErrors)
        {
            for (Double d : l) 
            {
                totalError += Math.abs(d);
            }
        }

        return totalError;
    }
    
    public void addBias(List<Double> outputs)
    {
        outputs.add(bias);
    }
    
    public void setBias(Double bias)
    {
        this.bias = bias;
    }
    
    public Layer getLayer(int index) 
    {
        
        return layers.get(index);
    }
}
