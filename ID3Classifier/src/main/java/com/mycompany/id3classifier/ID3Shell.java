/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.id3classifier;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *
 * @author Besseym
 */
public class ID3Shell {
     public static void main(String[] args) throws Exception {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("lensesData.csv");
            Instances dataSet = source.getDataSet();
            
            Discretize filter = new Discretize();
            filter.setInputFormat(dataSet);
            dataSet = Filter.useFilter(dataSet, filter);
            
            Standardize standardize = new Standardize();
            standardize.setInputFormat(dataSet);
            dataSet = Filter.useFilter(dataSet, standardize);
            
            dataSet.setClassIndex(dataSet.numAttributes() - 1);
            dataSet.randomize(new Random(9001)); //It's over 9000!!
            
            int folds = 10;
            //Perform crossvalidation
            Evaluation eval = new Evaluation(dataSet);
            for (int n = 0; n < folds; n++) 
            {
                int trainingSize = (int) Math.round(dataSet.numInstances() * .7);
                int testSize = dataSet.numInstances() - trainingSize;
            
                Instances trainingData = dataSet.trainCV(folds, n);
                Instances testData = dataSet.testCV(folds, n);
            
                ID3Classifier classifier = new ID3Classifier();
               // Id3 classifier = new Id3();
                classifier.buildClassifier(trainingData);
            
                 eval.evaluateModel(classifier, testData);
            }
            System.out.println(eval.toSummaryString("\nResults:\n", false));
	}
}

