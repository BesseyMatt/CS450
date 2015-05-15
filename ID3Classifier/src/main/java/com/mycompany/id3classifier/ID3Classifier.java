package com.mycompany.id3classifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Discretize;
import javafx.util.Pair;

import java.util.*;


/**
 * @author Besseym
 */
public class ID3Classifier extends Classifier 
{
    Node tree;

    /*
    * sameClass() : returns the if all instances are within the same class or not
    */
    private Pair<Boolean, Double> sameClass(List<Instance> instances) 
    {
        int classIndex = instances.get(0).classIndex();
        double tmpValue = Double.NaN;
        
        for (int i = 0; i < instances.size(); i++)
        {
            Instance instance = instances.get(i);
            
            // Assign value if we haven't yet
            if (Double.isNaN(tmpValue))
                tmpValue = instance.value(classIndex);
            
            else
            {
                double value = instance.value(classIndex);
                
                // only check if the data is there
                if (!Double.isNaN(value))
                    if (value != tmpValue)
                        // we've found multiple data values, not the same
                        return new Pair<>(false, Double.NaN);
            }
        }

        // only found 1 value for the class
        return new Pair<>(true, tmpValue);
    }
    
    /*
     * subset() : returns a lsit of instances that equal the given value
     */
    private List<Instance> subset(Map<Instance, Double> map, double value) 
    {
        ArrayList<Instance> list = new ArrayList<>();
    
        for (Instance instance : map.keySet()) 
            if (map.get(instance) == value) 
                list.add(instance);

        return list;
    }

    /*
    * valuesByAttribute() : Creates a map of the instances and splits them up according
    *                       to the value associated by that attribute.
    */
    private Map<Instance, Double> valuesByAttribute(List<Instance> instances, Attribute attribute) 
    {
        HashMap<Instance, Double> map = new HashMap<>();

        for (Instance instance : instances) 
            map.put(instance, instance.value(attribute));

        if (!attribute.isNominal())
            for (Instance key : map.keySet()) 
            {
                if (map.get(key) == Double.NaN) 
                {
                    //Do Nothing
                } 
                
                else if (map.get(key) < 0.5) 
                    map.put(key, 0.0); // 'Low'
                
                else
                    map.put(key, 1.0); // 'High'
            }

        return map;
    }

    /**
     SummarizeValues: returns how often each possible attribute value appears
     **/
    private Map<Double, Integer> summarizeValues(Map<Instance, Double> input) 
    {
        HashMap<Double, Integer> hashMap = new HashMap<>();

        for (Instance i : input.keySet()) 
        {
            if (!hashMap.containsKey(input.get(i)) || hashMap.get(i) == null)
                hashMap.put(input.get(i), 1);
            
            else
                hashMap.put(input.get(i), hashMap.get(i) + 1);
        }

        return hashMap;
    }

    /*
    * getMaxGain(): Looks at all possible information gains for each remaining
    *               attribute and returns the attribute with the best information
    *               gain in the given set.
    */
    public Attribute getMaxGain(List<Instance> instances, List<Attribute> attributes) 
    {
        Pair<Attribute, Double> maxGain = new Pair<>(null, Double.NEGATIVE_INFINITY);
        double totalEntropy = entropy(instances);
    
        for (Attribute attribute : attributes)
        {
            double tmpGain = gain(instances, attribute, totalEntropy);
            
            if (tmpGain > maxGain.getValue())
                maxGain = new Pair<>(attribute, tmpGain);
        }

        return maxGain.getKey();
    }

    /**
     * Entropy() using the machine learning entropy formula 
     */
     private double entropy(List<Instance> instances) 
    {
        double result = 0;
        Map<Double, Integer> summary = summarizeValues(valuesByAttribute(instances, instances.get(0).classAttribute()));
        
        for (Integer val : summary.values())
        {
            double proportion = val * 1.0 / instances.size();
            result -= proportion * Math.log(proportion) / Math.log(2);
        }

        return result;
    }
    
    /*
     * gain() : calculates the information gain for selecting the provide attribute 
     */ 
    private double gain(List<Instance> instances, Attribute attribute, double entropyOfSet) 
    {
        double gain = entropyOfSet;
        Map<Instance, Double> values = valuesByAttribute(instances, attribute);
        HashSet<Double> valueSet = new HashSet<>(values.values());
        
        for (Double d : valueSet) 
        {
            List<Instance> sub = subset(values, d);
            gain -= sub.size() * 1.0 / instances.size() * entropy(sub);
        }

        return gain;
    }

    /**
     * BuildTree() : Creates the tree recursively for all possible attributes
     */
    private Node buildTree(List<Instance> instances, List<Attribute> attributes) 
    {
        if (instances.size() < 1) 
            throw new UnsupportedOperationException("Error: Instances is empty");

        //returns if the members of instance are all the same class
        Pair<Boolean, Double> classification = sameClass(instances);
        
        //If all memebers have the same class create a leaf node and return
        if (classification.getKey()) 
            return new Node(classification.getValue());

        // if no attributes remain to use find the best class to return, calls
        // kNNClassifier.getClassification (to save code) and returns the most
        // common class
        if (attributes.isEmpty())
            return new Node(kNNClassifier.getClassification(instances));

        //Find what attribute will return the highest gain, this attribute
        //will be used to check next
        Attribute largestGain = getMaxGain(instances, attributes);
        Node n = new Node(instances, largestGain);
        
        Map<Instance, Double> vals =  valuesByAttribute(instances, largestGain);
        Map<Double, Integer> summary = summarizeValues(vals);
        
        //create new list of attributes minus the one we will be checking next
        ArrayList<Attribute> newList = new ArrayList<>(attributes);
        newList.remove(largestGain);
        
        //add enough children to this node for all keyset possiblities
        for (Double value : summary.keySet()) 
        {
            Node idNode = buildTree(subset(vals, value), newList);
            n.addChild(value, idNode);
        }

        return n;
    }

    /**
     * PrintTree(): Displays the contents of the tree, tabs are
     *              used to indicate a step down in the tree.
     */
    public void printTree(Node node, int level, Double value) 
    {
        //Root
        if (level == 0)
            System.out.println(node.attribute.name() + " -");
        
        else
        {
            //Add tab for each level down in the tree
            for (int i = 0; i < level; i++)
                System.out.print('\t');

            System.out.print(value);

            if (!node.isLeaf())
                System.out.print(" : " + node.getAttribute().name());

            System.out.print(" -");

            if (node.isLeaf()) 
                System.out.println(" " + node.leafValue);
 
            else
                System.out.println();
        }

        //Print the tree
        for (Node n : node.getChildren()) 
            printTree(n, level + 1, node.get(n));
    }

    /*
     * buildClassifier() : creates a list of instances and a list of attributes
     *                     then builds and displays the tree from this data.
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception 
    {
        List<Instance> instanceList = new ArrayList<>(instances.numInstances());
        
        for (int i = 0; i < instances.numInstances(); i++) 
            instanceList.add(instances.instance(i));

        List<Attribute> attributeList = new ArrayList<>(instances.numAttributes());
        
        for (int i = 0; i < instances.numAttributes(); i++) 
            if (i != instances.classIndex())
                attributeList.add(instances.attribute(i));

        tree = buildTree(instanceList, attributeList);
        printTree(tree, 0, 0.0);
    }

    /*
    * getClassification() : works down the tree and figures out what classification
    *                       should be returned. If data is missing the most common
    *                       data close to the passed value is returned.
    */
    public double getClassification(Instance instance, Node n) 
    {
        if (n.isLeaf()) 
            return n.leafValue;
        
        else 
        {
            Attribute attribute = n.getAttribute();
            
            if (Double.isNaN(instance.value(attribute))) 
            {
                Map<Double, Integer> classToCount = new HashMap<>();
                
                for (Node child : n.getChildren()) 
                {
                    Double value = getClassification(instance, child);
                    
                    if (!classToCount.containsKey(value) && classToCount.get(value) != null) 
                        classToCount.put(value, classToCount.get(value) + 1);
                    
                    else
                        classToCount.put(value, 1);
                }

                int maxCount = -1;
                double maxValue = 0;
                
                for (Double d : classToCount.keySet()) 
                    if (classToCount.get(d) > maxCount) 
                    {
                        maxCount = classToCount.get(d);
                        maxValue = d;
                    }

                return maxValue;
            } 
            
            else 
            {
                Double val = instance.value(n.getAttribute());
                if (val == null || Double.isNaN(val))
                    val = 0.0;

                // convert val to one of the 2 values we have doubles.
                if (!n.getAttribute().isNominal()) 
                {
                    if (val < 0.5) 
                        val = 0.0;
                    else 
                        val = 1.0;
                }

                Node child = n.get(val);
                if (child == null) 
                    //Calls the kNNClassifier's getClassification to save me 
                    //from writing it again
                    return kNNClassifier.getClassification(n.instances);
                
                else 
                    return getClassification(instance, n.get(val));
            }
        }
    }

    /*
     * classifyInstance() : This overrided function is called for each value to
     *                      be classified and looks at the tree and figures out
     *                      what classification should be returned.
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception
    {
        return getClassification(instance, tree);
    }
}

/*
 * Node Class: The node class is used to create the ID3 tree. Each node contains
 *             two maps (Node to double and double to Node) of all the child of 
 *             the node. The Node Class is really doing a lot of heavy lifting
 *             of this whole program and is essential to this program.
 */
class Node 
{
    List<Instance> instances;
    private Map<Node, Double> children = new HashMap<>();
    private Map<Double, Node> children2 = new HashMap<>();
    Attribute attribute;
    boolean isLeaf = false;
    double leafValue;

    public Node(List<Instance> instances, Attribute attribute) 
    {
        this.instances = instances;
        this.attribute = attribute;
    }

    public Node(double leafValue) 
    {
        isLeaf = true;
        this.leafValue = leafValue;
    }

    public boolean isLeaf() 
    {
        return isLeaf;
    }

    public Attribute getAttribute() 
    {
        return attribute;
    }

    public void addChild(Double value, Node n) 
    {
        children.put(n, value);
        children2.put(value, n);
    }

    public Double get(Node n) 
    {
        return children.get(n);
    }

    public Node get(Double d) 
    {
        if (children2.get(d) != null)
            return children2.get(d);
        
        else {
            Double closestDistanceKey = null;
            double closestDistance = Double.MAX_VALUE;
            
            for (Double key : children2.keySet()) {
                if (Math.abs(d - key) < closestDistance) {
                    closestDistance = Math.abs(d - key);
                    closestDistanceKey = key;
                }
            }
            
            return children2.get(closestDistanceKey);
        }
    }

    public Set<Node> getChildren() 
    {
        return children.keySet();
    }
}