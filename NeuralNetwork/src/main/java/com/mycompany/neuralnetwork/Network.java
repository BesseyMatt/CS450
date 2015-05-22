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
 * @author weatherwornwow
 */
public class Network {
   List<Layer> layers = new ArrayList<>();

   Double bias = 1.0;
   
    public Network(int inputCount, List<Integer> numberOfNeuronPerLayer) {
        
        if (numberOfNeuronPerLayer.isEmpty()) {
            throw new UnsupportedOperationException("numberOfNeuronPerLayer is empty");
        }

        // 1 extra for the bias
        layers.add(new Layer(numberOfNeuronPerLayer.get(0), inputCount + 1));

        for (int i = 1; i < numberOfNeuronPerLayer.size(); i++) {
            
            layers.add(new Layer(numberOfNeuronPerLayer.get(i),
                    numberOfNeuronPerLayer.get(i - 1) + 1));
        }
    }

    public List<Double> getOutputs(List<Double> inputs) {
        
        List<Double> outputs = new ArrayList<>(inputs);

        for (Layer layer : layers) {
            
            addBias(outputs);
            
            outputs = layer.produceOutputs(outputs);
        }

        return outputs;
    } 
    
    public void addBias(List<Double> outputs) {
        outputs.add(bias);
    }
    
    public void setBias(Double bias) {
        this.bias = bias;
    }
    
    public Layer getLayer(int index) {
        
        return layers.get(index);
    }
}
