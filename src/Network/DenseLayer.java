package Network;

import java.util.Arrays;
import java.util.function.Supplier;

/** A single layer of Neurons. Contains fully connected edges to every neuron in the previous layer */
public class DenseLayer extends Layer {

    /**
     * A 2D matrix of weights
     * <br>Rows: The neuron {@code n} in this layer
     * <br>Columns: The weight of a synapse pointing to {@code n}
     */ //made public for testing purposes, was protected
    public final double[][] weights;

    /**
     * A 2D matrix of weight velocities for SGD with momentum
     * <br>Rows: The neuron {@code n} in this layer
     * <br>Columns: The weight of a synapse pointing to {@code n}
     */ //made public for testing purposes, was protected
    public final double[][] weightsVelocity;

    /**
     * A 2D matrix of gradients of the loss function with respect to the weights
     * <br>Rows: The neuron {@code n} in this layer
     * <br>Columns: The weight of a synapse pointing to {@code n}
     */
    private final double[][] weightGradient;

    public DenseLayer(int nodesBefore, int nodes) {
        super(nodes);
        this.weights = new double[nodes][nodesBefore];
        this.weightsVelocity = new double[nodes][nodesBefore];
        this.weightGradient = new double[nodes][nodesBefore];
    }

    @Override
    public void initialize(Supplier<Double> initializer){
        super.initialize(initializer);
        for (int i = 0; i < nodes; i++)
            for (int j = 0; j < weights[0].length; j++)
                weights[i][j] = initializer.get();
    }

    /** Applies the weights and biases of this Layer to the given input. Returns a new array. */
    public double[] calculateWeightedOutput(double[] input) {
        double[] output = new double[nodes];

        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < input.length; j++) {
                assert Double.isFinite(weights[i][j]);
                output[i] += weights[i][j] * input[j];
            }
            output[i] += bias[i];
            assert Double.isFinite(output[i]) : "Weighted output has reached Infinity";
        }

        return output;
    }

    /**
     * Given the derivative array of the latest input sum,
     * calculates and shifts the given weight and bias gradients.
     * @return da_dC where a is the activation function of the layer before this one
     */
    public synchronized double[] updateGradient(double[] dz_dC, double[] x) {
        double[] da_dC = new double[weightGradient[0].length];
        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < weightGradient[0].length; j++) {
                assert Double.isFinite(weightGradient[i][j]);
                assert Double.isFinite(dz_dC[i]);
                assert Double.isFinite(x[j]);

                weightGradient[i][j] += dz_dC[i] * x[j];
                assert Double.isFinite(weightGradient[i][j]) : "weightGradient[i][j](" + weightGradient[i][j] + ") + dz_dC[i](" + dz_dC[i] + ") * x[j](" + x[j] + ") is equal to an invalid (Infinite) value";

                assert Double.isFinite(da_dC[j]);

                da_dC[j] += dz_dC[i] * weights[i][j];
            }
            assert Double.isFinite(biasGradient[i]);
            assert Double.isFinite(dz_dC[i]);

            biasGradient[i] += dz_dC[i];
        }
        return da_dC;
    }

    /**
     * Applies the {@code weightGradient} and {@code biasGradient} to the weight and bias of this java.Layer.
     * <br>Updates the weight and bias's gradient velocity vectors accordingly as well.
     */
    public void applyGradient(double adjustedLearningRate, double momentum, double beta, double epsilon) {
        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                assert Double.isFinite(weightGradient[i][j]);
                weightsVelocity[i][j] = beta * weightsVelocity[i][j] + (1 - beta) * (weightGradient[i][j] * weightGradient[i][j]);
                weights[i][j] -= adjustedLearningRate * weightGradient[i][j] / Math.sqrt(weightsVelocity[i][j] + epsilon);
            }
            assert Double.isFinite(biasGradient[i]);
            biasVelocity[i] = beta * biasVelocity[i] + (1 - beta) * (biasGradient[i] * biasGradient[i]);
            bias[i] -= adjustedLearningRate * biasGradient[i] / Math.sqrt(biasVelocity[i] + epsilon);
        }
    }

    public void clearGradient() {
        for (int i = 0; i < weightGradient.length; i++) weightGradient[i] = new double[weights[0].length];
        Arrays.fill(biasGradient, 0);
    }

    @Override
    public int getNumParameters() {
        return weights.length * weights[0].length + super.getNumParameters();
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Weights: ");
        Layer.ArraysDeepToString(weights, sb);
        sb.append("\nBiases: \n").append(Arrays.toString(bias));
        return sb.toString();
    }
}
