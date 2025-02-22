import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
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

    /** {@code vπ(s)}: Given a state s, returns the expected future reward of that state using the Markov Decision Process (MDP) */
    private static HashMap<Integer, Double> MDPValueFunction;

    private static final BiFunction<Integer, Integer, Double> stateToReward = (state, nextState) -> nextState == 1 ? -1 : (nextState == 1000 ? 1 : 0.0);

    /** Given a state, maps it to a binary function feature data, where the only non-zero value is a 1
     * at the index of where the state is located (Ex: State 230 means index at state 200 to 300 has value 1). */
    private static final Function<Integer, double[]> stateToFeatureMap = state -> {
        double[] featureMap = new double[numBuckets];
        featureMap[(state - 1) / (1000 / numBuckets)] = 1;
        return featureMap;
    };

    /** The Discount Rate Parameter of this particular problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    /** The learning rate used in stochastic gradient descent (SGD) */
    private static final double learningRate = 4e-5;

    static {
        MDPValueFunction = new HashMap<>();

        for (int state = 1; state <= 1000; state++)
            MDPValueFunction.put(state, 0.0);

        instantiateTrueValueFunction();
    }

    public static void main(String[] args) {
        for (int i = 0; i < 1_000_000; i++) {
            updateStepMC();
            updateEpisodeTD();
            System.out.println(i);
        }
        System.out.println("weightsMC: " + Arrays.toString(weightsMC));
        System.out.println("weightsTD: " + Arrays.toString(weightsTD));

        for (int i = 1; i <= 1000; i += 1000 / numBuckets) {
            double avg = 0;
            for (int j = 0; j < 1000 / numBuckets; j++)
                avg += MDPValueFunction.get(i + j);
            avg /= 1000.0 / numBuckets;
            System.out.println((i / (1000 / numBuckets))+": "+avg);
        }
    }

    /** Runs an episode of Thousand State Random Walking, since there
     * is no action to be taken this can be applied to any learning algorithm. */
    private static List<Pair<Integer, Double>> getEpisodeData() {
        ArrayList<Pair<Integer, Double>> stateRewards = new ArrayList<>();
        int currentState = 500;

        while (currentState != 1 && currentState != 1000) {

            int randomWalk = (int) (Math.random() * 200) - 100;
            if (randomWalk >= 0) randomWalk++;
            int nextState = Math.clamp(currentState + randomWalk, 1, 1000);

            stateRewards.add(new Pair<>(currentState, stateToReward.apply(currentState, nextState)));
            currentState = nextState;
        }
        return stateRewards;
    }

    /** Runs one episode of MC-algorithm and update accordingly using the update step */
    private static void updateStepMC() {
        List<Pair<Integer, Double>> stateRewards = getEpisodeData();
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
    private static void updateEpisodeTD() {
        int currentState = 500;

        while (currentState != 1 && currentState != 1000) {
            int randomWalk = (int) (Math.random() * 200) - 100;
            if (randomWalk >= 0) randomWalk++;
            int nextState = Math.clamp(currentState + randomWalk, 1, 1000);

            //update TD weights on learned values
            double[] currentStateFeatureMap = stateToFeatureMap.apply(currentState);
            double targetValue = stateToReward.apply(currentState, nextState) + (nextState == 1 || nextState == 1000 ? 0 : discountRate * LinAlg.dotProduct(weightsTD, stateToFeatureMap.apply(nextState)));
            weightsTD = LinAlg.add(weightsTD, LinAlg.scale(learningRate * (targetValue - LinAlg.dotProduct(weightsTD, currentStateFeatureMap)), currentStateFeatureMap));

            currentState = nextState;
        }
    }

    /** Instantiates {@link #MDPValueFunction} by calling {@link #updateMDPValue} on every state value until convergence */
    private static void instantiateTrueValueFunction() {
        for (int i = 0; i < 1000; i++) {
            HashMap<Integer, Double> newMDPValueFunction = new HashMap<>();
            for (int j = 2; j < 1000; j++)
                newMDPValueFunction.put(j, updateMDPValue(j));
            newMDPValueFunction.put(1, 0.0);
            newMDPValueFunction.put(1000, 0.0);
            MDPValueFunction = newMDPValueFunction;
        }
    }

    /** Runs an episode of MDP algorithm to update its value function */
    private static double updateMDPValue(int state) {
        double newValuation = 0, totalProbability = 0;

        int minState = Math.max(1, state - 100);
        double probability = (state - 100 < 1 ? (1 - (state - 100) + 1) * 0.005 : 0.005);
        totalProbability += probability;
        newValuation += probability * (stateToReward.apply(state, minState) + discountRate * MDPValueFunction.get(minState));

        int maxState = Math.min(1000, state + 100);
        probability = (state + 100 > 1000 ? (state + 100 - 1000 + 1) * 0.005 : 0.005);
        totalProbability += probability;
        newValuation += probability * (stateToReward.apply(state, maxState) + discountRate * MDPValueFunction.get(maxState));

        for (minState++; minState < state; minState++) {
            newValuation += 0.005 * (stateToReward.apply(state, minState) + discountRate * MDPValueFunction.get(minState));
            totalProbability += 0.005;
        }

        for (maxState--; maxState > state; maxState--) {
            newValuation += 0.005 * (stateToReward.apply(state, maxState) + discountRate * MDPValueFunction.get(maxState));
            totalProbability += 0.005;
        }

        assert Math.abs(totalProbability - 1) < 1e-5 : totalProbability;
        return newValuation;
    }
}
