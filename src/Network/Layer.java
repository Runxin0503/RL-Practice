package Network;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.Supplier;

public abstract class Layer {


    /** The number of Neurons in this layer */
    //made public for testing purposes, was protected
    public final int nodes;

    /** The bias of each neuron in this layer */
    //made public for testing purposes, was protected
    public final double[] bias;

    /** The bias velocity of each neuron in this layer, used in SGD with momentum */
    //made public for testing purposes, was protected
    public double[] biasVelocity;

    /** The bias velocity of each neuron in this layer, used in RMS-Prop */
    //made public for testing purposes, was protected
    public double[] biasVelocitySquared;

    /** The gradient of the bias with respect to the loss function */
    protected final double[] biasGradient;

    /** The number of times this Neural Network updated its weights and biases. */
    protected int t = 1;

    public Layer(int nodes) {
        this.nodes = nodes;
        this.bias = new double[nodes];
        this.biasGradient = new double[nodes];
    }

    /** Initializes the parameters of this Layer */
    void initialize(Supplier<Double> initializer, Optimizer optimizer) {
        if (optimizer == Optimizer.SGD_MOMENTUM || optimizer == Optimizer.ADAM)
            this.biasVelocity = new double[nodes];
        if (optimizer == Optimizer.RMS_PROP || optimizer == Optimizer.ADAM)
            this.biasVelocitySquared = new double[nodes];
        for (int i = 0; i < nodes; i++)
            bias[i] = initializer.get();
    }

    /** Passes the input through this Layer and returns a new array of numbers. */
    public abstract double[] calculateWeightedOutput(double[] input);

    /**
     * Given the derivative array of the latest input sum (dz_dC),
     * calculate and shift this layer's gradients.
     * @return da_dC where a is the activation function of the layer before this one
     */
    abstract double[] updateGradient(double[] dz_dC, double[] x);

    /**
     * Applies this layer's gradients to the parameters of this Layer.
     * <br>Updates the respective gradient velocity vectors accordingly as well.
     */
    void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        Consumer<Integer> updateRule;
        switch (optimizer){
            case SGD -> updateRule = i -> bias[i] -= adjustedLearningRate * biasGradient[i];
            case SGD_MOMENTUM -> updateRule = i -> {
                biasVelocity[i] = biasVelocity[i] * momentum + (1 - momentum) * biasGradient[i];
                bias[i] -= adjustedLearningRate * biasVelocity[i];
            };
            case RMS_PROP -> updateRule = i -> {
                biasVelocitySquared[i] = beta * biasVelocitySquared[i] + (1 - beta) * (biasGradient[i] * biasGradient[i]);
                bias[i] -= adjustedLearningRate * biasGradient[i] / Math.sqrt(biasVelocitySquared[i] + epsilon);
            };
            case ADAM -> {
                double correctionMomentum = 1 - Math.pow(momentum, t);
                double correctionBeta = 1 - Math.pow(beta, t);
                updateRule = i -> {
                    biasVelocity[i] = momentum * biasVelocity[i] + (1 - momentum) * biasGradient[i];
                    biasVelocitySquared[i] = beta * biasVelocitySquared[i] + (1 - beta) * biasGradient[i] * biasGradient[i];
                    double correctedVelocity = biasVelocity[i] / correctionMomentum;
                    double correctedVelocitySquared = biasVelocitySquared[i] / correctionBeta;
                    bias[i] -= adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon);
                    assert Double.isFinite(bias[i]) : "\ncorrectedVelocity: " + correctedVelocity + "\ncorrectedVelocitySquared: " + correctedVelocitySquared + "\nbiasVelocity: " + biasVelocity[i] + "\nbiasVelocitySquared: " + biasVelocitySquared[i];
                };
                t++;
            }
            case null, default -> throw new IllegalStateException("Unexpected value: " + optimizer);
        }
        for (int i = 0; i < nodes; i++) {
            assert Double.isFinite(biasGradient[i]);
            updateRule.accept(i);
        }
    }

    /** Clears this layer's gradient for its parameters with respect to the loss function */
    abstract void clearGradient();

    /** Returns the number of learnable parameters in this layer */
    public int getNumParameters() {
        return bias.length;
    }

    @Override
    public abstract String toString();

    @Override
    public abstract Object clone();

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Layer o)) return false;
        return nodes == o.nodes &&
                Arrays.equals(bias, o.bias) &&
                Arrays.equals(biasVelocity, o.biasVelocity) &&
                Arrays.equals(biasVelocitySquared, o.biasVelocitySquared) &&
                Arrays.equals(biasGradient, o.biasGradient);
    }

    /** A helper method for subclasses of Layer to use in their {@link #toString()} methods.
     * <br>Unlike {@link Arrays#deepToString(Object[])}, this method truncates all weight numbers to 2 digits,
     * making visualization easier and less clustered. */
    static void ArraysDeepToString(double[][] array, StringBuilder sb) {
        for (int i = 0; i < array.length; i++) {
            sb.append("[");
            for (int j = 0; j < array[i].length; j++) {
                sb.append(String.format("%.2f", array[i][j]));
                if (j < array[i].length - 1)
                    sb.append(", ");
            }
            sb.append("]");
            if (i < array.length - 1)
                sb.append(",");
            sb.append("\n");
        }
    }
}
