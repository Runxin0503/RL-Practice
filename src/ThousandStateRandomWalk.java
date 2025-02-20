import java.util.function.Function;

/**
 * A 1000-state random walk example as an evaluation task for RL algorithms in function approximation.
 * Each state is given as an integer from 1 to 1000.<br>
 * There is no actions in this example, each step has a 50% chance of moving to any state.<br>
 * 100 left of the current state (chosen uniformly), same with moving to right states.<br>
 * Start at state = 500 and end at state 1 or 1000, gaining rewards of -1 and 1 respectively.
 */
public class ThousandStateRandomWalk {

    /** The number of buckets (features) to have, more features generally means more accurate function approximation
     * but this value should be fixed at 10 or intervals of 10. */
    private static final int numBuckets = 10;

    /** Weights of the Monte-Carlo value function, used to approximate the actual true value function */
    private static final double[] weightsMC = new double[numBuckets];

    /** Weights of the 1-step Temporal Difference value function, used to approximate the actual true value function */
    private static final double[] weightsTD = new double[numBuckets];

    /** Given a state, maps it to a binary function feature data, where the only non-zero value is a 1
     * at the index of where the state is located (Ex: State 230 means index at state 200 to 300 has value 1). */
    private static final Function<Integer,double[]> stateToFeatureMap = state -> {
        double[] featureMap = new double[numBuckets];
        featureMap[state / (1000/numBuckets)] = 1;
        return featureMap;
    };

    /** The learning rate used in stochastic gradient descent (SGD) */
    private static final double learningRate = 0.1;

    public static void main(String[] args) {
        //todo implement
    }

    /** Runs one episode of MC-algorithm and update accordingly using the update step */
    public static void updateStepMC(){
        //todo implement
    }

    /** Runs one episode of TD-algorithm and updates accordingly using the update step */
    public static void updateEpisodeTD(){
        //todo implement
    }
}
