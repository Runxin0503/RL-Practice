package Network;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/** Activation Function enum containing regular and derivative functions of commonly-used Activation Functions */
public enum Activation {
    none(input -> {
        double[] output = new double[input.length];
        System.arraycopy(input,0,output,0,input.length);
        return output;
    },(input,gradient) -> {
        double[] output = new double[input.length];
        //input will always be 1 with respect to output
        System.arraycopy(gradient, 0, output, 0, input.length);
        return output;
    }),
    ReLU(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (input[i] > 0 ? input[i] : 0);
        return output;
    }, (input,gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = gradient[i] * (input[i] > 0 ? 1.0 : 0);
        return output;
    }),
    sigmoid(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = 1 / (1 + Math.exp(-input[i]));
        return output;
    }, (input,gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++){
            double a = 1/(1+Math.exp(-input[i]));
            output[i] = gradient[i] * a * (1-a);
        }
        return output;
    }),
    tanh(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
            output[i] = Math.tanh(input[i]);
        return output;
    },(input,gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double tanhValue = Math.tanh(input[i]);
            output[i] = gradient[i] * (1-tanhValue*tanhValue);
        }
        return output;
    }),
    LeakyReLU(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] > 0 ? input[i] : 0.1 * input[i];
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        for(int i = 0; i < input.length; i++) output[i] = gradient[i] * (input[i] > 0 ? 1.0 : 0.1);
        return output;
    }),
    softmax(input -> {
        double[] output = new double[input.length];
        double latestInputSum = 0,max = Double.MIN_VALUE;
        for (double num : input) max = Math.max(max, num);
        for (double num : input) latestInputSum += Math.exp(num - max);
        for (int i = 0; i < input.length; i++) output[i] = Math.exp(input[i] - max) / latestInputSum;
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        double[] softmaxOutput = new double[input.length];
        double latestInputSum = 0,max = Double.MIN_VALUE;
        for (double num : input) max = Math.max(max, num);
        for (double num : input) latestInputSum += Math.exp(num - max);
        for (int i = 0; i < input.length; i++) softmaxOutput[i] = Math.exp(input[i] - max) / latestInputSum;

        // Compute the gradient using the vectorized form
        double dotProduct = 0.0;
        for (int i = 0; i < softmaxOutput.length; i++) {
            dotProduct += softmaxOutput[i] * gradient[i];
        }

        for (int i = 0; i < softmaxOutput.length; i++) {
            output[i] = softmaxOutput[i] * (gradient[i] - dotProduct);
        }

        return output;
    });

    private static final Random RANDOM = new Random();
    private static final BiFunction<Integer,Integer,Double> HE_Initialization = (inputSize,outputSize)->RANDOM.nextGaussian(0,Math.sqrt(2.0/(inputSize+outputSize)));
    private static final BiFunction<Integer,Integer,Double> XAVIER_Initialization = (inputSize,outputSize)->RANDOM.nextGaussian(0,Math.sqrt(1/Math.sqrt(inputSize+outputSize)));


    private final Function<double[],double[]> function;
    private final BiFunction<double[],double[],double[]> derivativeFunction;
    Activation(Function<double[],double[]> function,BiFunction<double[],double[],double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /** Returns the result of AF(x) for every x in {@code input} array*/
    public double[] calculate(double[] input) {
        for(double v : input) assert Double.isFinite(v) : "Attempted to input invalid values into Activation Function " + Arrays.toString(input);
        double[] output = this.function.apply(input);
        for(double v : output) assert Double.isFinite(v) : "Activation Function returning invalid values " + Arrays.toString(input) + "\n" + Arrays.toString(output);
        return output;
    }

    /**
     * Effect: multiplies each element in {@code da_dC[i]} with their corresponding element {@code AF'(z[i])}
     * @return {@code dz_dC}
     */
    public double[] derivative(double[] z, double[] da_dC) {
        for(double v : da_dC) assert Double.isFinite(v) : "Attempted to input invalid values into Deriv of Activation Function " + Arrays.toString(z) + "  " + Arrays.toString(da_dC);
        double[] newGradient = this.derivativeFunction.apply(z, da_dC);
        for(double v : newGradient) assert Double.isFinite(v) : "Deriv of Activation Function returning invalid values " + Arrays.toString(z) + "  " + Arrays.toString(da_dC) + "\n" + Arrays.toString(newGradient);
        return newGradient;
    }

    /**
     * Returns the weights and bias initializer supplier best associated with {@code AF} function
     */
    public static Supplier<Double> getInitializer(Activation AF,int inputNum,int outputNum){
        if(AF.equals(ReLU) || AF.equals(LeakyReLU)) return ()->HE_Initialization.apply(inputNum,outputNum);
        else return ()->XAVIER_Initialization.apply(inputNum,outputNum);
    }
}
