package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Heriberto Gonzalez, ML asgt 6
 */

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 *
 * @author dkauchak
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	//The actual surrogate loss function to use
	private int lossFunction;
	private int regularizationType;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	//Variable Hyperparameters
	private double learningRate = 0.01;
	private double regularizerWeight = 0.01;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	protected int iterations = 10;

	/**
	 * Zero param constructor
	 */
	public GradientDescentClassifier() {
	}

	/**
	 * Selects the loss function to use
	 *
	 * @param lossType - the loss type to use in calculations for GradientDescent
	 */
	public void setLoss(int lossType) {
		assert lossType == EXPONENTIAL_LOSS || lossType == HINGE_LOSS : "Illegal Loss Type";
		lossFunction = lossType;
	}

	/**
	 * Selects the regularization function to use
	 *
	 * @param regularizationType - the regularization type to use in calculations for GradientDescent
	 */
	public void setRegularization(int regularizationType) {
		assert regularizationType == NO_REGULARIZATION ||
				regularizationType == L1_REGULARIZATION ||
				regularizationType == L2_REGULARIZATION : "Illegal Regularization Type";
		this.regularizationType = regularizationType;
	}

	/**
	 * Set the regularizerWeight to be lambda
	 *
	 * @param lambda
	 */
	public void setLambda(double lambda) {
		regularizerWeight = lambda;
	}

	/**
	 * Set the learning rate to be eta
	 *
	 * @param eta
	 */
	public void setEta(double eta) {
		learningRate = eta;
	}

	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 *
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 *
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 *
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();

		//for each iteration...
		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);

			//for each example...
			for (Example e : training) {
				makeAdjustments(e);
			}
		}
	}

	/**
	 * Make necessary adjustments to weights and bias based on an example
	 *
	 * @param e
	 */
	private void makeAdjustments(Example e) {
		double label = e.getLabel();
		double distanceFromHyperplane = getDistanceFromHyperplane(e, weights, b);

		//get the loss function and the value given the label and distance from hyperplane
		double loss = getLoss(label, distanceFromHyperplane);

		//For each weight, indexed by the feature set of the example...
		for (Integer featureIndex : e.getFeatureSet()) {
			//get the direction to update in
			double updateDirection = label * e.getFeature(featureIndex);
			//get the regularization for this weight
			double regularization = getRegularization(weights.get(featureIndex));
			//adjust the weight based on the objective function
			weights.put(featureIndex, weights.get(featureIndex) + learningRate * ((updateDirection * loss) - (regularizerWeight * regularization)));
		}
		double bRegularization = getRegularization(b);
		//update b - same equation as line above, but update direction just becomes label (since Xij becomes 1)
		b = b + learningRate * ((label * loss) - (regularizerWeight * bRegularization));
	}

	/**
	 * Decide on the regularization function via setRegularization and calculate it
	 *
	 * @param weight - the current weight being updated
	 * @return
	 */
	private double getRegularization(double weight) {
		//L1 regularization is the sign of wj
		if (regularizationType == L1_REGULARIZATION) {
			if (weight > 0) return 1;
			else if (weight < 0) return -1;
			return 0;
			//L2 regularization is just wj
		} else if (regularizationType == L2_REGULARIZATION) {
			return weight;
		}
		//No regularization
		return 0;
	}

	/**
	 * Decide on the loss function based on which was set with setLoss and calculate loss based on input
	 *
	 * @param label                  - label of the example
	 * @param distanceFromHyperplane - distance of the example from the hyperplane
	 * @return
	 */
	private double getLoss(double label, double distanceFromHyperplane) {
		if (lossFunction == EXPONENTIAL_LOSS) {
			return Math.pow(Math.E, -label * distanceFromHyperplane);
		} else if (lossFunction == HINGE_LOSS) {
			return label * distanceFromHyperplane < 1 ? 1 : 0;
		}
		throw new IllegalArgumentException("Illegal Loss Function found while getting loss");
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}

	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}


	/**
	 * Get the prediction from the current set of weights on this example
	 *
	 * @param e the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 *
	 * @param e      example to predict
	 * @param w      the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		//for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}
}