import Network.Activation;
import Network.Cost;
import Network.NN;
import org.junit.jupiter.api.RepeatedTest;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class WindyHighwayTest {

    private static final double learningRate = 1e-4;
    private static NN Policy;
    private static final double epsilon = 1e-3;

    @RepeatedTest(1000)
    void test() {
        Policy = new NN.NetworkBuilder().setInputNum(2).setHiddenAF(Activation.LeakyReLU)
                .setOutputAF(Activation.softmax).setCostFunction(Cost.crossEntropy)
                .addDenseLayer(40).addDenseLayer(20)
                .addDenseLayer(3).build();
        System.out.println(Arrays.toString(Policy.calculateOutput(new double[]{0, 0})));
        for(int i=0;i<1000;i++){
            double x = 0,y = 0;
            double targetValue = 5;
            int actionIndex = 0;

            double[] output = Policy.calculateOutput(new double[]{x,y});
            double adjustedGradientMultiplier = learningRate * targetValue / (output[actionIndex] + epsilon);
            if(adjustedGradientMultiplier >= 0) {
                output = new double[3];
                output[actionIndex] = 1;
            }
            else{
                output[actionIndex] = 0;
                double sum = output[0] + output[1] + output[2];
                output[0] /= sum;
                output[1] /= sum;
                output[2] /= sum;
            }
            NN.learn(Policy,Math.abs(adjustedGradientMultiplier),0,0,0,new double[][]{{x,y}},new double[][]{output});
        }
        System.out.println(Arrays.toString(Policy.calculateOutput(new double[]{0, 0})));
        assertTrue(Policy.calculateOutput(new double[]{0, 0})[0]>0.9);
    }

    @RepeatedTest(30)
    void test2() {
        WindyHighway.main(new String[0]);
    }
}
