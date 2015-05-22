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
public class Layer {
    
    List<Neuron> neurons = new ArrayList<>();

    public Layer(int neuronCount, int inputCount) {
        
        for (int i = 0; i < neuronCount; i++) {
            
            neurons.add(new Neuron(inputCount));
        }
    }

    public List<Double> produceOutputs(List<Double> inputs) {
        
        List<Double> outputs = new ArrayList<>();
        
        for (Neuron neuron : neurons) {
        
            outputs.add(neuron.produceOutput(inputs));
        }

        return outputs;
    }
    
    public Neuron getNeuron(int index) {
        
        return neurons.get(index);
    }
}
