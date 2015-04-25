package com.mycompany.experimentshell;


import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Besseym
 */
public class Main {
    public static void main(String[] args) throws Exception {
            DataSource source = new DataSource("irisData.csv");
            Instances dataSet = source.getDataSet();
            
            dataSet.setClassIndex(dataSet.numAttributes() - 1);
            
            dataSet.randomize(new Random(1));
            
            int trainingSize = (int) Math.round(dataSet.numInstances() * .7);
            int testSize = dataSet.numInstances() - trainingSize;
            
            Instances trainingData = new Instances(dataSet, 0, trainingSize);
            Instances testData = new Instances(dataSet, trainingSize, testSize);
            
            HardCodedClassifier classifier = new HardCodedClassifier();
            classifier.buildClassifier(trainingData);
            
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier, testData);
            
            System.out.println(eval.toSummaryString("\nResults:\n", false));
	}
}
