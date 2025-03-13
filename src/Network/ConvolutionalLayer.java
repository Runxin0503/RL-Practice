package Network;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/** A Collection of Kernels and output neurons specialized in Image Processing.
 * <br><br> A Kernel is a 2D matrix of synapses that slides over the input matrix,
 * summing the multiplication of input and weights to a single output neuron. This
 * sliding motion creates a 2D output matrix, where each neuron in the output correspond
 * to the result of running the kernel on a subsection of the input matrix.
 * <br>When the input matrix is 3D, each layer of the input matrix is processed independently
 * and the 3D output matrix is compressed into 2D matrix through addition.
 * <br><br>Output Dimension: The Output layer will be a 3D matrix, where each kernel creates a layer of its own.
 * The dimension of each kernel layer is:
 * <br> -WIDTH = ceilDiv(inputWidth - kernelWidth + 1, strideWidth)
 * <br> -HEIGHT = ceilDiv(inputHeight - kernelHeight + 1, strideHeight)
 * <br><br>Requires: {@code inputWidth * inputHeight * inputLength} = {@code input.length} in {@link #calculateWeightedOutput(double[])}
 */
public class ConvolutionalLayer extends Layer {

    /**
     * An array of kernels, which each are a 2D matrix of weights
     * <br>Layers: The kernel at that layer.
     * <br>Rows & Columns: A 2D collection of weights connecting from the previous layer to
     * a single neuron {@code n} in this layer.
     */
    private final double[][][] kernels;

    /**
     * An array of kernel velocities, which each are a 2D matrix of weights velocities used in SGD with momentum
     * <br>Layers: The kernel at that layer.
     * <br>Rows & Columns: A 2D collection of weights connecting from the previous layer to
     * a single neuron {@code n} in this layer.
     */
    private double[][][] kernelsVelocity;

    /**
     * An array of kernel velocities, which each are a 2D matrix of weights velocities for RMS-Prop
     * <br>Layers: The kernel at that layer.
     * <br>Rows & Columns: A 2D collection of weights connecting from the previous layer to
     * a single neuron {@code n} in this layer.
     */
    private double[][][] kernelsVelocitySquared;

    /**
     * An array of kernel gradients, which each are a 2D matrix of gradients of the loss function with respect to the weights
     * <br>Layers: The kernel at that layer.
     * <br>Rows & Columns: A 2D collection of weights connecting from the previous layer to
     * a single neuron {@code n} in this layer.
     */
    private final double[][][] kernelsGradient;

    private final int inputWidth, inputHeight, inputLength;
    private final int kernelWidth, kernelHeight, numKernels;
    private final int outputWidth,outputHeight;
    private final int strideWidth, strideHeight;
    private final boolean padding;
    private final int[][][] inputVectorToInputMatrix;

    public ConvolutionalLayer(int inputWidth, int inputHeight, int inputLength,
                              int kernelWidth, int kernelHeight, int numKernels,
                              int strideWidth, int strideHeight, boolean padding) {
        super(padding ? inputWidth * inputHeight * numKernels :
                Math.ceilDiv(inputWidth - kernelWidth + 1, strideWidth) *
                        Math.ceilDiv(inputHeight - kernelHeight + 1, strideHeight) *
                        numKernels);
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputLength = inputLength;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.numKernels = numKernels;
        this.strideWidth = strideWidth;
        this.strideHeight = strideHeight;
        this.outputWidth = Math.ceilDiv(inputWidth - kernelWidth + 1, strideWidth);
        this.outputHeight = Math.ceilDiv(inputHeight - kernelHeight + 1, strideHeight);
        this.padding = padding;
        final int paddingWidth,paddingHeight;
        if (padding) {
            paddingWidth = inputWidth * strideWidth - strideWidth - inputWidth + kernelWidth;
            paddingHeight = inputHeight * strideHeight - strideHeight - inputHeight + kernelHeight;
        } else {
            paddingWidth = outputWidth * strideWidth + kernelWidth - inputWidth;
            paddingHeight = outputHeight * strideHeight + kernelHeight - inputHeight;
        }

        this.kernels = new double[numKernels][kernelWidth][kernelHeight];
        this.kernelsGradient = new double[numKernels][kernelWidth][kernelHeight];

        //initialize inputVectorToInputMatrix converter and find padding
        inputVectorToInputMatrix = new int[inputWidth + paddingWidth][inputHeight + paddingHeight][inputLength];
        int paddingLeft, paddingUp;
        if (padding) { //preserve original dimension size
            paddingLeft = Math.ceilDiv(paddingWidth, 2);
            paddingUp = Math.ceilDiv(paddingHeight, 2);
        } else { //only pad when stride is too big
            paddingLeft = paddingWidth;
            paddingUp = paddingHeight;
        }

        //transform 1D input array into 3D input matrix and add padding
        for (int layer = 0; layer < inputLength; layer++)
            for (int x = 0; x < inputWidth + paddingWidth; x++) {
                int i;
                if (x < paddingLeft) i = paddingLeft - x;
                else if (x >= inputWidth + paddingLeft) i = 2 * inputWidth + paddingLeft - x;
                else i = x - paddingLeft;
                for (int y = 0; y < inputHeight + paddingHeight; y++) {
                    int j;
                    if (y < paddingUp) j = paddingUp - y;
                    else if (y >= inputHeight + paddingUp) j = 2 * inputHeight + paddingUp - y;
                    else j = y - paddingUp;
                    inputVectorToInputMatrix[x][y][layer] = inputWidth * inputHeight * layer + inputWidth * j + i;
                }
            }
    }

    @Override
    void initialize(Supplier<Double> initializer,Optimizer optimizer) {
        super.initialize(initializer, optimizer);

        if(optimizer == Optimizer.SGD_MOMENTUM || optimizer == Optimizer.ADAM)
            this.kernelsVelocity = new double[numKernels][kernelWidth][kernelHeight];
        if(optimizer == Optimizer.RMS_PROP || optimizer == Optimizer.ADAM)
            this.kernelsVelocitySquared = new double[numKernels][kernelWidth][kernelHeight];

        for (int i = 0; i < kernelWidth; i++)
            for (int j = 0; j < kernelHeight; j++)
                for (int k = 0; k < numKernels; k++)
                    kernels[k][i][j] = initializer.get();
    }

    /** Applies the weights and biases of this Layer to the given input. Returns a new array. */
    public double[] calculateWeightedOutput(double[] input) {
        assert inputWidth * inputHeight * inputLength == input.length;

        //use kernels to scan through each layer of input matrix, create output matrix
        double[] output = new double[outputWidth * outputHeight * numKernels];
        IntStream.range(0, numKernels).parallel().forEach(kernel -> {
            for (int layer = 0; layer < inputLength; layer++)
                for (int x = 0; x < outputWidth; x++)
                    for (int y = 0; y < outputHeight; y++) {
                        //loop kernel through each kernel-region to completely populate a location in the output
                        double weightedSum = 0;
                        for (int scanX = 0; scanX < kernelWidth; scanX++)
                            for (int scanY = 0; scanY < kernelHeight; scanY++)
                                weightedSum += kernels[kernel][scanX][scanY] * input[inputVectorToInputMatrix[x * strideWidth + scanX][y * strideHeight + scanY][layer]];

                        int nodeAbsPos = x + y * outputWidth + kernel * outputWidth * outputHeight;
                        output[nodeAbsPos] = weightedSum + bias[nodeAbsPos];
                    }
        });

        return output;
    }

    /**
     * Given the derivative array of the latest input sum,
     * calculates and shifts the given weight and bias gradients.
     * @return da_dC where a is the activation function of the layer before this one
     */
    @Override
    double[] updateGradient(double[] dz_dC, double[] x) {
        double[] da_dC = new double[inputWidth * inputHeight * inputLength];

        IntStream.range(0, numKernels).parallel().forEach(kernel -> {
            IntStream.range(0,inputLength).parallel().forEach(layer -> {
                for (int i = 0; i < outputWidth; i++)
                    for (int j = 0; j < outputHeight; j++) {
                        int index = i + j * outputWidth + kernel * outputWidth * outputHeight;
                        assert Double.isFinite(dz_dC[index]);
                        for (int kernelX = 0; kernelX < kernelWidth; kernelX++)
                            for (int kernelY = 0; kernelY < kernelHeight; kernelY++) {
                                int absXPos = inputVectorToInputMatrix[i * strideWidth + kernelX][j * strideHeight + kernelY][layer];
                                assert Double.isFinite(kernelsGradient[kernel][kernelX][kernelY]);
                                assert Double.isFinite(x[absXPos]);


                                kernelsGradient[kernel][kernelX][kernelY] += dz_dC[index] * x[absXPos];
                                da_dC[absXPos] += dz_dC[index] * kernels[kernel][kernelX][kernelY];
                            }
                    }
            });
        });

        return da_dC;
    }

    /**
     * Applies the {@code weightGradient} and {@code biasGradient} to the weight and bias of this Layer.
     * <br>Updates the weight and bias's gradient velocity vectors accordingly as well.
     */
    @Override
    void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        IntStream.range(0, numKernels).parallel().forEach(layer -> {
            BiConsumer<Integer,Integer> updateRule;
            switch (optimizer){
                case SGD -> updateRule = (x,y) -> kernels[layer][x][y] -= adjustedLearningRate * kernelsGradient[layer][x][y];
                case SGD_MOMENTUM -> updateRule = (x,y) -> {
                    kernelsVelocity[layer][x][y] = kernelsVelocity[layer][x][y] * momentum + (1 - momentum) * kernelsGradient[layer][x][y];
                    kernels[layer][x][y] -= adjustedLearningRate * kernelsVelocity[layer][x][y];
                };
                case RMS_PROP -> updateRule = (x,y) -> {
                    kernelsVelocitySquared[layer][x][y] = beta * kernelsVelocitySquared[layer][x][y] + (1 - beta) * (kernelsGradient[layer][x][y] * kernelsGradient[layer][x][y]);
                    kernels[layer][x][y] -= adjustedLearningRate * kernelsGradient[layer][x][y] / Math.sqrt(kernelsGradient[layer][x][y] + epsilon);
                };
                case ADAM -> {
                    double correctionMomentum = 1 - Math.pow(momentum, t);
                    double correctionBeta = 1 - Math.pow(beta, t);
                    updateRule = (x,y) -> {
                        kernelsVelocity[layer][x][y] = momentum * kernelsVelocity[layer][x][y] + (1 - momentum) * kernelsGradient[layer][x][y];
                        kernelsVelocitySquared[layer][x][y] = beta * kernelsVelocitySquared[layer][x][y] + (1 - beta) * kernelsGradient[layer][x][y] * kernelsGradient[layer][x][y];
                        double correctedVelocity = kernelsVelocity[layer][x][y] / correctionMomentum;
                        double correctedVelocitySquared = kernelsVelocitySquared[layer][x][y] / correctionBeta;
                        kernels[layer][x][y] -= adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon);
                        assert Double.isFinite(kernels[layer][x][y]) : "\ncorrectedVelocity: " + correctedVelocity + "\ncorrectedVelocitySquared: " + correctedVelocitySquared + "\nweightsVelocity: " + kernelsVelocity[x][y][layer] + "\nweightsVelocitySquared: " + kernelsVelocitySquared[x][y][layer];                    };
                }
                case null, default -> throw new IllegalStateException("Unexpected value: " + optimizer);
            }

            for (int x = 0; x < kernelWidth; x++)
                for (int y = 0; y < kernelHeight; y++) {
                    assert Double.isFinite(kernelsGradient[layer][x][y]);
                    updateRule.accept(x,y);
                }
        });

        super.applyGradient(optimizer,adjustedLearningRate,momentum,beta,epsilon);
    }

    @Override
    void clearGradient() {
        for (int i = 0; i < kernels.length; i++)
            kernelsGradient[i] = new double[kernels[0].length][kernels[0][0].length];
        Arrays.fill(biasGradient, 0);
    }

    @Override
    public int getNumParameters() {
        return kernels.length * kernels[0].length * kernels[0][0].length + super.getNumParameters();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<numKernels;i++){
            sb.append("Kernel ").append(i).append(":\n");
            Layer.ArraysDeepToString(kernels[i],sb);
            sb.append('\n');
        }
        sb.append("Biases: \n").append(Arrays.toString(bias));
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if(!(obj instanceof ConvolutionalLayer o) || super.equals(obj)) return false;

        return inputWidth == o.inputWidth && inputHeight == o.inputHeight && inputLength == o.inputLength &&
                kernelWidth == o.kernelWidth && kernelHeight == o.kernelHeight && numKernels == o.numKernels &&
                outputWidth == o.outputWidth && outputHeight == o.outputHeight &&
                strideWidth == o.strideWidth && strideHeight == o.strideHeight &&
                padding == o.padding && Arrays.deepEquals(kernels, o.kernels) &&
                Arrays.deepEquals(kernelsVelocity, o.kernelsVelocity) &&
                Arrays.deepEquals(kernelsVelocitySquared, o.kernelsVelocitySquared) &&
                Arrays.deepEquals(kernelsGradient, o.kernelsGradient) &&
                Arrays.deepEquals(inputVectorToInputMatrix, o.inputVectorToInputMatrix);
    }

    @Override
    public Object clone() {
        ConvolutionalLayer newLayer = new ConvolutionalLayer(inputWidth,inputHeight,inputLength,kernelWidth,kernelHeight,numKernels,strideWidth,strideHeight,padding);
        System.arraycopy(bias, 0, newLayer.bias, 0, nodes);
        if(!Objects.isNull(biasVelocity)) {
            newLayer.biasVelocity = new double[biasVelocity.length];
            newLayer.kernelsVelocity = new double[kernelsVelocity.length][kernelsVelocity[0].length][kernelsVelocity[0][0].length];
            System.arraycopy(biasVelocity, 0, newLayer.biasVelocity, 0, nodes);
        }
        if(!Objects.isNull(biasVelocitySquared)) {
            newLayer.biasVelocitySquared = new double[biasVelocitySquared.length];
            newLayer.kernelsVelocitySquared = new double[kernelsVelocitySquared.length][kernelsVelocitySquared[0].length][kernelsVelocitySquared[0][0].length];
            System.arraycopy(biasVelocitySquared, 0, newLayer.biasVelocitySquared, 0, nodes);
        }
        System.arraycopy(biasGradient, 0, newLayer.biasGradient, 0, nodes);
        for(int i=0;i<kernels.length;i++) for(int j=0;j<kernels[0].length;j++){
            System.arraycopy(kernels[i][j],0,newLayer.kernels[i][j],0,kernels[0].length);
            if(!Objects.isNull(kernelsVelocity)) System.arraycopy(kernelsVelocity[i][j],0,newLayer.kernelsVelocity[i][j],0,kernels[0].length);
            if(!Objects.isNull(kernelsVelocitySquared))System.arraycopy(kernelsVelocitySquared[i][j],0,newLayer.kernelsVelocitySquared[i][j],0,kernels[0].length);
            System.arraycopy(kernelsGradient[i][j],0,newLayer.kernelsGradient[i][j],0,kernels[0].length);
        }

        return newLayer;
    }
}
