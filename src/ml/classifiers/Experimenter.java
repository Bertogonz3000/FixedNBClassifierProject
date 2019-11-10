package ml.classifiers;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

import java.util.*;

/**
 * Berto Gonzalez and Sarah Bashir ASGT 7 Nov.3rd 2019
 */

public class Experimenter {

    public static void main(String[] args) {
        DataSet dataSet = new DataSet("/Users/berto/Desktop/Pomona4thyr/ml/homeworkSeven/data/wines.train", DataSet.TEXTFILE);

//        experimentOne(dataSet);
//        experimentTwo(dataSet);
        experimentFour(dataSet);
    }

    private static void experimentOne(DataSet dataSet) {
        NBClassifier classifier = new NBClassifier();
        DataSetSplit splitData = dataSet.split(0.8);
        DataSet trainingData = splitData.getTrain();
        ArrayList<Example> testingData = splitData.getTest().getData();

        classifier.setUseOnlyPositiveFeatures(true);
        classifier.train(trainingData);

        for (double i = 0.0; i < 0.15; i += 0.01) {
            classifier.setLambda(i);
            double avgAccuracy = 0.0;
            avgAccuracy += calculateAccuracy(classifier, testingData);
            System.out.println(avgAccuracy);
        }
    }

    private static void experimentTwo(DataSet dataSet) {
        NBClassifier classifier = new NBClassifier();
        classifier.setUseOnlyPositiveFeatures(true);
        ClassifierTimer.timeClassifier(classifier, dataSet, 5);
        System.out.println("\n");

    }

    private static void experimentFour(DataSet dataSet) {
        DataSetSplit data = dataSet.split(0.8);
        CrossValidationSet crossSet = new CrossValidationSet(data.getTrain(), 10);
        NBClassifier classifier = new NBClassifier();
        HashMap<Double, Double> predictions = new HashMap<>();


        DataSetSplit crossSplit = crossSet.getValidationSet(1);
        DataSet trainData = crossSplit.getTrain();
        DataSet testData = crossSplit.getTest();
        classifier.train(trainData);

        for (Example example : testData.getData()) {
            double labelPrediction = classifier.classify(example);
            double confidence = classifier.confidence(example);

            predictions.put(labelPrediction, confidence);
        }

        ArrayList<double[]> sortedLabelsWithConfidence = new ArrayList<>();

        for (int k = 0; k < predictions.size(); k++) {
            double labelWithBiggestConfidence = Collections.max(predictions.entrySet(), Comparator.comparingDouble(Map.Entry::getValue)).getKey();
            double confidence = predictions.get(labelWithBiggestConfidence);
            double[] labelAndConfidence = new double[2];
            labelAndConfidence[0] = labelWithBiggestConfidence;
            labelAndConfidence[1] = confidence;
            sortedLabelsWithConfidence.add(labelAndConfidence);
            predictions.remove(labelWithBiggestConfidence);
        }



        for (double[] labelAndConfidence : sortedLabelsWithConfidence) {
            double currentConfidenceThreshold = labelAndConfidence[1];
            double accuracyWithThreshold = calculateAccuracyWithThreshold(classifier, testData.getData(), currentConfidenceThreshold);
            System.out.println(currentConfidenceThreshold);
            System.out.println(accuracyWithThreshold);
            System.out.println("\n");
        }


    }

    private static double calculateAccuracy(Classifier classifier, ArrayList<Example> examples) {
        int correct = 0;
        int total = 0;
        //For each example
        for (Example e : examples) {
            double classification = classifier.classify(e);
            if (classification == e.getLabel()) {
                correct++;
            }
            total++;
        }
        double accuracy = (double) correct / (double) total;
        return accuracy;
    }

    private static double calculateAccuracyWithThreshold(Classifier classifier, ArrayList<Example> examples, double threshold) {
        int correct = 0;
        int total = 0;

        for (Example example : examples) {
            double classification = classifier.classify(example);
            if (classifier.confidence(example) >= threshold) {
                if (classification == example.getLabel()) {
                    correct++;
                }
                total++;
            }
        }
        double accuracy = (double) correct / (double) total;
        return accuracy;
    }
}
