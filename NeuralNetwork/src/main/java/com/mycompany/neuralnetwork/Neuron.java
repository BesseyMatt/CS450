/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author Besseym
 */
public class Neuron {
    
     List<Double> weights = new ArrayList<>();

    public Neuron(int inputCount) {
        double oneOver = 1.0 / Math.sqrt(inputCount);
        
        for (int i = 0; i < inputCount; i++) {
            weights.add(Math.random() * 2.0 * oneOver - oneOver);
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

        return 1 / (1 + Math.exp(-sum));
    }
    
    public Double getWeight(int index) {
        
        return weights.get(index);
    }
}
