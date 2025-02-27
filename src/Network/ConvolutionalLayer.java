package Network;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class ConvolutionalLayer extends Layer {

    private final double[][][] kernels;

    private final double[][][] kernelsVelocity;

    private final double[][][] kernelGradient;

    private final int inputWidth, inputHeight, inputLength;
    private final int kernelWidth, kernelHeight, numKernels;
    private final int outputWidth,outputHeight;
    private final int strideWidth, strideHeight;
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
        final int paddingWidth,paddingHeight;
        if (padding) {
            paddingWidth = inputWidth * strideWidth - strideWidth - inputWidth + kernelWidth;
            paddingHeight = inputHeight * strideHeight - strideHeight - inputHeight + kernelHeight;
        } else {
            paddingWidth = outputWidth * strideWidth + kernelWidth - inputWidth;
            paddingHeight = outputHeight * strideHeight + kernelHeight - inputHeight;
        }

        this.kernels = new double[numKernels][kernelWidth][kernelHeight];
        this.kernelsVelocity = new double[numKernels][kernelWidth][kernelHeight];
        this.kernelGradient = new double[numKernels][kernelWidth][kernelHeight];

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
    public void initialize(Supplier<Double> initializer) {
        for (int i = 0; i < kernelWidth; i++)
            for (int j = 0; j < kernelHeight; j++)
                for (int k = 0; k < numKernels; k++)
                    kernels[k][i][j] = initializer.get();
    }

    /** Applies the weights and biases of this java.Layer to the given input. Returns a new array. */
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
    public double[] updateGradient(double[] dz_dC, double[] x) {
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
                                assert Double.isFinite(kernelGradient[kernel][kernelX][kernelY]);
                                assert Double.isFinite(x[absXPos]);


                                kernelGradient[kernel][kernelX][kernelY] += dz_dC[index] * x[absXPos];
                                da_dC[absXPos] += dz_dC[index] * kernels[kernel][kernelX][kernelY];
                            }
                    }
            });
        });

        return da_dC;
    }

    /**
     * Applies the {@code weightGradient} and {@code biasGradient} to the weight and bias of this java.Layer.
     * <br>Updates the weight and bias's gradient velocity vectors accordingly as well.
     */
    @Override
    public void applyGradient(double adjustedLearningRate, double beta, double epsilon) {
        IntStream.range(0, kernelWidth).parallel().forEach(x -> {
            for (int y = 0; y < kernelHeight; y++)
                for (int layer = 0; layer < numKernels; layer++) {
                    assert Double.isFinite(kernelGradient[layer][x][y]);
                    kernelsVelocity[layer][x][y] = beta * kernelsVelocity[layer][x][y] + (1 - beta) * (kernelGradient[layer][x][y] * kernelGradient[layer][x][y]);
                    kernels[layer][x][y] -= adjustedLearningRate * kernelGradient[layer][x][y] / Math.sqrt(kernelsVelocity[layer][x][y] + epsilon);
                }
        });
        IntStream.range(0, bias.length).parallel().forEach(i -> {
            assert Double.isFinite(biasGradient[i]);
            biasVelocity[i] = beta * biasVelocity[i] + (1 - beta) * (biasGradient[i] * biasGradient[i]);
            bias[i] -= adjustedLearningRate * biasGradient[i] / Math.sqrt(biasVelocity[i] + epsilon);
        });
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
    public int getNumParameters() {
        return kernels.length * kernels[0].length * kernels[0][0].length + super.getNumParameters();
    }

    @Override
    public void clearGradient() {
        for (int i = 0; i < kernels.length; i++)
            kernelGradient[i] = new double[kernels[0].length][kernels[0][0].length];
        Arrays.fill(biasGradient, 0);
    }
}
