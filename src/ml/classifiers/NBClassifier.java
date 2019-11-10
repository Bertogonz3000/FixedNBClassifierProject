package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

import java.awt.*;
import java.util.*;

/**
 * Berto Gonzalez and Sarah Bashir ASGT 7 Nov. 3, 2019
 */

/**
 * This class uses naive bayes probabilities to learn and classify data
 */
public class NBClassifier implements Classifier {

    //Lambda
    private double smoothingParam = 0.01;

    private boolean useOnlyPositiveFeatures = false;

    //Hashmap containing the counts of each label, must call getLabelCounts to do this
    private HashMapCounter<Double> labelCounts = new HashMapCounter<Double>();
    private HashMap<Double, HashMapCounter<Integer>> featureCountsByLabel = new HashMap<>();

    private DataSet dataSet;

    /**
     * Zero param constructor
     */
    public NBClassifier() {

    }

    /**
     * Train the model on the given dataset
     *
     * @param data
     */
    @Override
    public void train(DataSet data) {
        this.dataSet = data;
        getLabelCounts(data);
        getAllFeatureCountsForLabels();
    }

    /**
     * Classify the example argument based on the trained model
     *
     * @param example
     * @return - label of this example with highest probability.
     */
    @Override
    public double classify(Example example) {
        HashMap<Double, Double> logProbs = new HashMap<>();
        //For each label in the dataset, calculate the log probability and throw it into a hashmap with its label
        for (Double label : dataSet.getLabels()) {
            logProbs.put(label, getLogProb(example, label));
        }
        return Collections.max(logProbs.entrySet(), Comparator.comparingDouble(Map.Entry::getValue)).getKey();
    }

    /**
     * Return the probability of the most probable label for this example
     *
     * @param example
     * @return - probability of most likely label for this example
     */
    @Override
    public double confidence(Example example) {
        HashMap<Double, Double> logProbs = new HashMap<>();
        //For each label in the dataset, calculate the log probability and throw it into a hashmap with its label
        for (Double label : dataSet.getLabels()) {
            logProbs.put(label, getLogProb(example, label));
        }
        //return probability of this example
        return Collections.max(logProbs.values());
    }

    /**
     * return the log (base 10 ) probability of the example with the label under
     * the current trained model
     *
     * @param ex
     * @param label
     * @return - the log prob of the label for this example
     */
    public double getLogProb(Example ex, double label) {
        //Start the running example probability with the label probability = p(Y)
        double runningExampleProb = Math.log(labelCounts.get(label) / (double) dataSet.getLabels().size());
        //return log probability using either all the features or just those in the example.
        return useOnlyPositiveFeatures ? getOnlyPositiveProbability(ex, label, runningExampleProb) : getMathmaticallyCorrectProbability(ex, label, runningExampleProb);
    }

    /**
     * One of two ways to get probability, this one uses only positive and present features
     *
     * @return - log prob using only positive features
     */
    private double getOnlyPositiveProbability(Example ex, double label, double runningExampleProb) {
        //For each featureIndex, get the probability for that feature with this label, p(Xi|Y)
        for (Integer featureIndex : ex.getFeatureSet()) {
            double featureProb = Math.log(getFeatureProb(featureIndex, label));
            runningExampleProb += featureProb;
        }
        //return the running probability = p(Y) * prodOfAll(p(Xi|y))
        return runningExampleProb;
    }

    /**
     * One of two ways to get probability, this one is the mathmatically correct way to calculate Bayes
     *
     * @return - log prob using all features
     */
    private double getMathmaticallyCorrectProbability(Example ex, double label, double runningExampleProb) {
        //Set of all features in the example
        Set<Integer> exampleFeatureSet = ex.getFeatureSet();
        //for each feature in the dataSet...
        for (Integer featureIndex : dataSet.getAllFeatureIndices()) {
            //if feature is in example, multiply by regular prob
            if (exampleFeatureSet.contains(featureIndex)) {
                runningExampleProb += Math.log(getFeatureProb(featureIndex, label));
            } else {
                //if not, multiply by 1-regular prob
                runningExampleProb += Math.log(1 - getFeatureProb(featureIndex, label));
            }
        }
        return runningExampleProb;
    }

    /**
     * give p(Xi|Y) for the model
     *
     * @param featureIndex
     * @param label
     * @return - get the probability that this an example has this feature
     */
    public double getFeatureProb(int featureIndex, double label) {
        //count(Xi,y)
        double featureCount = featureCountsByLabel.get(label).get(featureIndex);
        //(count(Xi,y) * lambda) / (count(y) + possibleXis * lambda)
        return (featureCount + smoothingParam) / (labelCounts.get(label) + (2 * smoothingParam));
    }

    /**
     * Get the count of examples with label 'label' that have feature 'featureIndex'
     *
     * @param featureIndex
     * @param label
     * @return - count all instances of this feature in this label
     */
    private int getFeatureCountForLabel(int featureIndex, double label) {
        int featureCount = 0;
        for (Example example : dataSet.getData()) {
            //if word is in sentence and label the one we're looking for
            if (example.getLabel() == label && example.getFeatureSet().contains(featureIndex)) {
                //increment word count
                featureCount++;
            }
        }
        return featureCount;
    }

    /**
     * Count all feature counts for each label
     */
    private void getAllFeatureCountsForLabels() {
        //for each label...
        for (Double label : dataSet.getLabels()) {
            //for each feature...
            for (Integer featureIndex : dataSet.getAllFeatureIndices()) {
                //if the hashmap doesn't already have this label as a key...
                if (!featureCountsByLabel.containsKey(label)) {
                    //make a new hashmap counter and increment it to the feature count for this label for this feature
                    HashMapCounter<Integer> featureCount = new HashMapCounter<>();
                    featureCount.increment(featureIndex, getFeatureCountForLabel(featureIndex, label));
                    featureCountsByLabel.put(label, featureCount);
                } else {
                    //otherwise just incrememnt
                    featureCountsByLabel.get(label).increment(featureIndex, getFeatureCountForLabel(featureIndex, label));
                }
            }
        }
    }

    /**
     * Get the probability of any one label in the label set
     *
     */
    private void getLabelCounts(DataSet data) {
        //Count all instances of each label in the dataset and place into global hashmap
        for (Example example : data.getData()) {
            labelCounts.increment(example.getLabel());
        }
    }

    /**
     * Set the regularization/smoothing parameter
     *
     * @param newLambda
     */
    public void setLambda(double newLambda) {
        smoothingParam = newLambda;
    }

    /**
     * Set the boolean that decides on which classification variant to use
     *
     * @param use
     */
    public void setUseOnlyPositiveFeatures(boolean use) {
        useOnlyPositiveFeatures = use;
    }
}
