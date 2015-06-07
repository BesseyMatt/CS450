/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Besseym
 */
public class NeuralNetworkClassifier extends Classifier
{
    Network network;
    int layers;
    int iterations;
    double learningFactor;

    public NeuralNetworkClassifier(int layers, int iterations, double learningFactor) {
        this.layers = layers;
        this.iterations = iterations;
        this.learningFactor = learningFactor;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        int inputCount = instances.numAttributes() - 1;

        List<Integer> nodesPerLayer = new ArrayList<>();

        for (int i = 0; i < layers - 1; i++) {
            nodesPerLayer.add(inputCount); 
        }

        nodesPerLayer.add(instances.numDistinctValues(instances.classIndex()));

        network = new Network(inputCount, nodesPerLayer);

        ArrayList<Double> errorsPerIteration = new ArrayList<>();
        for (int j = 0; j < iterations; j++) {
            double errorsPer = 0;
            for (int k = 0; k < instances.numInstances(); k++) {
                Instance instance = instances.instance(k);

                
                List<Double> input = new ArrayList<>();
                for (int i = 0; i < instance.numAttributes(); i++) 
                {
                    if (Double.isNaN(instance.value(i)) && i != instance.classIndex())
                        input.add(0.0);
                    else if (i != instance.classIndex())
                        input.add(instance.value(i));
                }

                errorsPer += network.train(input, instance.value(instance.classIndex()), learningFactor);
            }
            
            errorsPerIteration.add(errorsPer);
           
        }
        
         //Display Errors This is used to collect the data for the graph 
        //for (Double d : errorsPerIteration) 
        //{
          //  System.out.println(d);
        //}
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception 
    {
    
        List<Double> input = new ArrayList<>();
    
        for (int i = 0; i < instance.numAttributes(); i++) 
        {  
            if (Double.isNaN(instance.value(i)) && i != instance.classIndex())
                input.add(0.0);
            
            else if (i != instance.classIndex())
                input.add(instance.value(i));
        }
        
        List<Double> outputs = network.getOutputs(input);
        
        double largeVal = -1;
        int index = 0;
        
        for (int i = 0; i < outputs.size(); i++)
        {
            double temp = outputs.get(i);
            
            if (temp > largeVal) 
            {
                largeVal = temp;
                index = i;
            }
        }
        
        return index;
    }
    
}
