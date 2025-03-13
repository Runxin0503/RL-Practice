package Network;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BiConsumer;
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
    public double[][] weightsVelocity;

    /**
     * A 2D matrix of weight velocities for RMS-Prop
     * <br>Rows: The neuron {@code n} in this layer
     * <br>Columns: The weight of a synapse pointing to {@code n}
     */ //made public for testing purposes, was protected
    public double[][] weightsVelocitySquared;

    /**
     * A 2D matrix of gradients of the loss function with respect to the weights
     * <br>Rows: The neuron {@code n} in this layer
     * <br>Columns: The weight of a synapse pointing to {@code n}
     */
    private final double[][] weightsGradient;

    public DenseLayer(int nodesBefore, int nodes) {
        super(nodes);
        this.weights = new double[nodes][nodesBefore];
        this.weightsGradient = new double[nodes][nodesBefore];
    }

    @Override
    void initialize(Supplier<Double> initializer,Optimizer optimizer){
        super.initialize(initializer,optimizer);

        if(optimizer == Optimizer.SGD_MOMENTUM || optimizer == Optimizer.ADAM)
            this.weightsVelocity = new double[nodes][weights[0].length];
        if(optimizer == Optimizer.RMS_PROP || optimizer == Optimizer.ADAM)
            this.weightsVelocitySquared = new double[nodes][weights[0].length];

        for (int i = 0; i < nodes; i++)
            for (int j = 0; j < weights[0].length; j++)
                weights[i][j] = initializer.get();
    }

    /** Applies the weights and biases of this Layer to the given input. Returns a new array. */
    @Override
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
    @Override
    synchronized double[] updateGradient(double[] dz_dC, double[] x) {
        double[] da_dC = new double[weightsGradient[0].length];
        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < weightsGradient[0].length; j++) {
                assert Double.isFinite(weightsGradient[i][j]);
                assert Double.isFinite(dz_dC[i]);
                assert Double.isFinite(x[j]);

                weightsGradient[i][j] += dz_dC[i] * x[j];
                assert Double.isFinite(weightsGradient[i][j]) : "weightsGradient[i][j](" + weightsGradient[i][j] + ") + dz_dC[i](" + dz_dC[i] + ") * x[j](" + x[j] + ") is equal to an invalid (Infinite) value";

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
     * Applies the {@code weightsGradient} and {@code biasGradient} to the weight and bias of this Layer.
     * <br>Updates the weight and bias's gradient velocity vectors accordingly as well.
     */
    @Override
    void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        BiConsumer<Integer,Integer> updateRule;
        switch (optimizer){
            case SGD -> updateRule = (i,j) -> weights[i][j] -= adjustedLearningRate * weightsGradient[i][j];
            case SGD_MOMENTUM -> updateRule = (i,j) -> {
                weightsVelocity[i][j] = weightsVelocity[i][j] * momentum + (1 - momentum) * weightsGradient[i][j];
                weights[i][j] -= adjustedLearningRate * weightsVelocity[i][j];
            };
            case RMS_PROP -> updateRule = (i,j) -> {
                weightsVelocitySquared[i][j] = beta * weightsVelocitySquared[i][j] + (1 - beta) * (weightsGradient[i][j] * weightsGradient[i][j]);
                weights[i][j] -= adjustedLearningRate * weightsGradient[i][j] / Math.sqrt(weightsVelocitySquared[i][j] + epsilon);
            };
            case ADAM -> {
                double correctionMomentum = 1 - Math.pow(momentum, t);
                double correctionBeta = 1 - Math.pow(beta, t);
                updateRule = (i,j) -> {
                    weightsVelocity[i][j] = momentum * weightsVelocity[i][j] + (1 - momentum) * weightsGradient[i][j];
                    weightsVelocitySquared[i][j] = beta * weightsVelocitySquared[i][j] + (1 - beta) * weightsGradient[i][j] * weightsGradient[i][j];
                    double correctedVelocity = weightsVelocity[i][j] / correctionMomentum;
                    double correctedVelocitySquared = weightsVelocitySquared[i][j] / correctionBeta;
                    weights[i][j] -= adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon);
                    assert Double.isFinite(weights[i][j]) : "\ncorrectedVelocity: " + correctedVelocity + "\ncorrectedVelocitySquared: " + correctedVelocitySquared + "\nweightsVelocity: " + weightsVelocity[i][j] + "\nweightsVelocitySquared: " + weightsVelocitySquared[i][j];
                };
            }
            case null, default -> throw new IllegalStateException("Unexpected value: " + optimizer);
        }

        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                assert Double.isFinite(weightsGradient[i][j]);
                updateRule.accept(i,j);
            }
        }

        super.applyGradient(optimizer,adjustedLearningRate,momentum,beta,epsilon);
    }

    void clearGradient() {
        for (int i = 0; i < weightsGradient.length; i++) weightsGradient[i] = new double[weights[0].length];
        Arrays.fill(biasGradient, 0);
    }

    @Override
    public int getNumParameters() {
        return weights.length * weights[0].length + super.getNumParameters();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Weights: ").append(Arrays.deepToString(weights));
        Layer.ArraysDeepToString(weights, sb);
        sb.append("\nBiases: \n").append(Arrays.toString(bias));
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if(!(obj instanceof DenseLayer o) || super.equals(obj)) return false;
        return Arrays.deepEquals(weights, o.weights) &&
                Arrays.deepEquals(weightsVelocity,o.weightsVelocity) &&
                Arrays.deepEquals(weightsVelocitySquared,o.weightsVelocitySquared) &&
                Arrays.deepEquals(weightsGradient,o.weightsGradient);
    }

    @Override
    public Object clone() {
        int nodesBefore = weights[0].length;
        DenseLayer newLayer = new DenseLayer(nodesBefore,nodes);
        System.arraycopy(bias, 0, newLayer.bias, 0, nodes);
        if(!Objects.isNull(biasVelocity)) {
            newLayer.biasVelocity = new double[biasVelocity.length];
            newLayer.weightsVelocity = new double[weightsVelocity.length][weightsVelocity[0].length];
            System.arraycopy(biasVelocity, 0, newLayer.biasVelocity, 0, nodes);
        }
        if(!Objects.isNull(biasVelocitySquared)) {
            newLayer.biasVelocitySquared = new double[biasVelocitySquared.length];
            newLayer.weightsVelocitySquared = new double[weightsVelocitySquared.length][weightsVelocitySquared[0].length];
            System.arraycopy(biasVelocitySquared, 0, newLayer.biasVelocitySquared, 0, nodes);
        }
        System.arraycopy(biasGradient, 0, newLayer.biasGradient, 0, nodes);
        for(int i=0;i<weights.length;i++) {
            System.arraycopy(weights[i], 0, newLayer.weights[i], 0, nodesBefore);
            if(!Objects.isNull(weightsVelocity)) System.arraycopy(weightsVelocity[i], 0, newLayer.weightsVelocity[i], 0, nodesBefore);
            if(!Objects.isNull(weightsVelocitySquared)) System.arraycopy(weightsVelocitySquared[i], 0, newLayer.weightsVelocitySquared[i], 0, nodesBefore);
            System.arraycopy(weightsGradient[i], 0, newLayer.weightsGradient[i], 0, nodesBefore);
        }
        return newLayer;
    }
}
