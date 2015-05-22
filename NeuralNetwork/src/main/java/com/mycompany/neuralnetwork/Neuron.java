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
public class Neuron {
    
     List<Double> weights = new ArrayList<>();

    public Neuron(int inputCount) {
        for (int i = 0; i < inputCount; i++) {
            weights.add(Math.random() * 2.0 - 1.0);
        }
    }

    public double produceOutput(List<Double> inputs) {
        if (inputs.size() != weights.size()) {
            throw new UnsupportedOperationException("Incorrect Number Of Inputs. Expected "
                + weights.size() + " and received " + inputs.size());
        }

        double sum = 0;
        for (int i = 0; i < weights.size(); i++) {
            sum += weights.get(i) * inputs.get(i);
        }

        return (sum <= 0 ? 0 : 1);
    }
    
    public Double getWeight(int index) {
        
        return weights.get(index);
    }
}
