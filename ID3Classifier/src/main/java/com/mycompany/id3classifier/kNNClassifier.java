/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.id3classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
/**
 *
 * @author Besseym
 */
public class kNNClassifier extends Classifier {
    
    Instances saved;
    final int k;
    Boolean bool;
    
    public kNNClassifier(int k) {
        this.k = k;
        bool = false;
    }
    
    public static double getClassification(List<Instance> instances) {
        
        int index = instances.get(0).classIndex();
        HashMap<Double, Integer> counts = new HashMap<>();
        
        for (Instance instance : instances) {
            double val = instance.value(index);
            
            if (!counts.containsKey(val))
                counts.put(val, 1);
            
            else {
                counts.put(val, counts.get(val) + 1);
            }        
        }
        
        int maxCount = 0;
        double maxValue = 0;
        
        for (Entry<Double, Integer> entry : counts.entrySet()) {
            
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                maxValue = entry.getKey();
            }      
        }
        
        return maxValue;
    }
    
    private static double findDistance(Instance instance1, Instance instance2) {
        double total = 0;
        int totalAttributes = instance1.numAttributes();
        for (int i = 0; i < totalAttributes; i++) {
            if (instance1.classIndex() == i)
                continue;
            
            double difference = 0;
            
            if (instance1.attribute(i).isNumeric()) {
                difference = Math.abs(instance1.value(i) - instance2.value(i));
            }
            
            else {
                if (!instance1.stringValue(i).equals(instance2.stringValue(i))) {
                    difference = 1;
                }
            }
            
            total += Math.pow(difference, totalAttributes);
        }
        
        return Math.pow(total, 1.0/totalAttributes);
    }
    
    @Override
    public void buildClassifier(Instances data) {
        saved = new Instances(data);
    }
    
    @Override
    public double classifyInstance(Instance instance) {
         
        //Distance between all saved Instances from the passed Instance
        HashMap<Instance, Double> map = new HashMap<Instance, Double>();
        
        for (int i = 0; i < saved.numInstances(); i++) {
            Instance temp = saved.instance(i);
            map.put(temp, findDistance(temp, instance));
        }

        ArrayList<Entry<Instance, Double>> sorted = new ArrayList<>(map.entrySet());
        Collections.sort(sorted, new Comparator<Entry<Instance, Double>>() {
            @Override
            public int compare(Entry<Instance, Double> e1, Entry<Instance, Double> e2) 
            {
                return e1.getValue().compareTo(e2.getValue());
            }
        });

        List<Instance> kNearest = new ArrayList<Instance> ();

        for (Entry<Instance, Double> inst : sorted) {
            kNearest.add(inst.getKey());

            if (kNearest.size() >= k) {
                break;
            }
        }

        return getClassification(kNearest);
    }
}
