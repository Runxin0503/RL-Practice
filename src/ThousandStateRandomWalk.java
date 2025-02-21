import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.BiFunction;
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
    private static double[] weightsMC = new double[numBuckets];

    /** Weights of the 1-step Temporal Difference value function, used to approximate the actual true value function */
    private static double[] weightsTD = new double[numBuckets];

    private static final BiFunction<Integer,Integer, Double> stateToReward = (state,nextState) -> nextState <= 1 ? -1 : (nextState >= 1000 ? 1 : 0.0);

    /** Given a state, maps it to a binary function feature data, where the only non-zero value is a 1
     * at the index of where the state is located (Ex: State 230 means index at state 200 to 300 has value 1). */
    private static final Function<Integer, double[]> stateToFeatureMap = state -> {
        double[] featureMap = new double[numBuckets];
        featureMap[(state-1) / (1000 / numBuckets)] = 1;
        return featureMap;
    };

    /** The Discount Rate Parameter of this particular problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    /** The learning rate used in stochastic gradient descent (SGD) */
    private static final double learningRate = 4e-5;

    public static void main(String[] args) {
        for(int i=0;i<100_000;i++){
            updateStepMC();
//            System.out.println(weightsMC[0]);
        }
        for(int i : new int[]{1,101,201,301,401,501,601,701,801,901}){
            System.out.println(LinAlg.dotProduct(stateToFeatureMap.apply(i),weightsMC));
        }
    }

    /** Runs one episode of MC-algorithm and update accordingly using the update step */
    public static void updateStepMC() {
        ArrayList<Pair<Integer, Double>> stateRewards = new ArrayList<>();
        int currentState = 500;

        while (currentState > 1 && currentState < 1000) {

            int randomWalk = (int)(Math.random() * 200) - 100;
            if(randomWalk >= 0) randomWalk++;
            int nextState = currentState + randomWalk;

            stateRewards.add(new Pair<>(currentState, stateToReward.apply(currentState,nextState)));
            currentState = nextState;
        }

        //tempWeightsMC stores the weights before processing the episode so the weights in the equation
        // will not change when updating
        double[] tempWeightsMC = weightsMC;
        for (int i = 0; i < stateRewards.size(); i++) {
            double targetValue = 0;
            for (int j = stateRewards.size() - 1; j >= i; j--) {
                targetValue = discountRate * targetValue + stateRewards.get(j).second();
            }

            double[] stateFeatureMap = stateToFeatureMap.apply(stateRewards.get(i).first());
            weightsMC = LinAlg.add(weightsMC, LinAlg.scale(learningRate * (targetValue - LinAlg.dotProduct(tempWeightsMC, stateFeatureMap)), stateFeatureMap));
        }
    }

    /** Runs one episode of TD-algorithm and updates accordingly using the update step */
    public static void updateEpisodeTD() {
        //todo implement
    }
}
