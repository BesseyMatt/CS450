/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.knnclassifier;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *
 * @author Besseym
 */
public class kNNShell {
     public static void main(String[] args) throws Exception {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("carData.csv");
            Instances dataSet = source.getDataSet();
            
            Standardize standardize = new Standardize();
            standardize.setInputFormat(dataSet);
            dataSet = Filter.useFilter(dataSet, standardize);
            
            dataSet.setClassIndex(dataSet.numAttributes() - 1);
            dataSet.randomize(new Random(9001)); //It's over 9000!!
            
            int trainingSize = (int) Math.round(dataSet.numInstances() * .7);
            int testSize = dataSet.numInstances() - trainingSize;
            
            Instances trainingData = new Instances(dataSet, 0, trainingSize);
            Instances testData = new Instances(dataSet, trainingSize, testSize);
            
            kNNClassifier classifier = new kNNClassifier(3);
            classifier.buildClassifier(trainingData);
            
            //Used to compare to Weka's built in KNN algorithm
            //Classifier classifier = new IBk(1);
            //classifier.buildClassifier(trainingData);
            
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier, testData);
            
            System.out.println(eval.toSummaryString("\nResults:\n", false));
	}
}
