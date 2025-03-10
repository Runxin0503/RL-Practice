import Network.Activation;
import Network.Cost;
import Network.NN;
import Utils.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * A Windy Highway simulation that trains an agent based on policy-gradient methods.
 * <br>State consists of a coordinate value x,y ~ [0,1]
 * <br>Action consists of 3 actions where each action has some noise
 * <br>After taking an action, the current horizontal position determines the strength of the wind pushing the agent's
 * x-coordinate and affects their next state.
 * <br>The reward of each state-transition is determined by the next state's horizontal position.
 */
public class WindyHighway {

    /** Given state transition, returns the reward. */
    private static final BiFunction<Pair<State, Action>, State, Double> stateTransitionToReward = (stateActionPair, nextState) -> Math.sin(nextState.x * 2 * Math.PI);

    /** Given the current state, returns the wind strength blowing to the right at that point. */
    private static final Function<State, Double> stateToWindValue = state -> Math.cos(state.x * 2 * Math.PI) * 0.02;

    /** {@code π(s)}: Given State s, returns the probability of taking either of the 3 valid actions in {@link Action} */
    public static NN Policy;

    /** {@code vπ(s)}: Given State s, returns the expected future returns assuming {@link #Policy} is followed.
     * <br>Used as a baseline function in REINFORCE algorithm. */
    public static NN valueFunction;

    /** The Discount Rate Parameter of the WindyHighway problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    /** {@code alpha (α)}, otherwise known as the step size, controls the learning rate of the policy and how fast it converges. */
    private static final double learningRate = 1e-5;

    /** Similar to {@link #learningRate}, except used in value function learning.
     * <br>Value function must learn faster than Policy itself, so this learning rate must be higher than {@link #learningRate}. */
    private static final double valueFunctionLearningRate = 1e-3;

    /** The parameter controlling the exploratory nature of {@link #Policy}.
     * <br>Higher the temperature, higher the Policy will explore and vice versa. */
    private static final double temperature = 5;

    public static void runTest() {
        main(new String[0]);
    }

    public static void main(String[] args) {
        Policy = new NN.NetworkBuilder().setInputNum(2).setHiddenAF(Activation.LeakyReLU)
                .setOutputAF(Activation.softmax).setCostFunction(Cost.crossEntropy)
                .addDenseLayer(40).addDenseLayer(20)
                .addDenseLayer(3).setTemperature(temperature).build();
        valueFunction = new NN.NetworkBuilder().setInputNum(2).setHiddenAF(Activation.LeakyReLU)
                .setOutputAF(Activation.none).setCostFunction(Cost.diffSquared)
                .addDenseLayer(40).addDenseLayer(20).addDenseLayer(1)
                .setTemperature(1).build();

        for (int i = 0; i < 10_000; i++) {
            updatePolicyOnEpisode();
//            System.out.println(i);
        }
        runEpisode().forEach(System.out::println);
        System.out.println("Evaluation: " + evaluatePolicyOnEpisode());
        updatePolicyOnEpisode();
        System.out.println("Evaluation: " + evaluatePolicyOnEpisode());
    }

    /** Uses the current {@link #Policy} to run 1,000 episodes and returns its average reward over those episodes */
    private static double evaluatePolicyOnEpisode() {
        double reward = 0;
        int iterations = 1000;

        //turn off exploration
        Policy.setTemperature(1);
        for (int i = 0; i < iterations; i++) {
            List<Pair<Pair<State, Action>, Double>> data = runEpisode();
            for (Pair<Pair<State, Action>, Double> pair : data) {
                reward += pair.second();
            }
        }
        Policy.setTemperature(temperature);

        return reward / iterations;
    }

    /**
     * Runs an episode of WindyHighway to collect data, then execute the following algorithm to update policy for every state action pair (s,a):
     * <br>1. Calculate the future returns of reward discounted over time (targetValue) at state s after taking action a.
     * <br>2. Calculate the advantage value by subtracting vπ(s) from targetValue
     * <br>3. Calls {@link NN#learn} on value function to update it using the targetValue
     * <br>4. Adjusted-learning-rate = learningRate * discountRate^t * targetValue / π(s,a)
     * <br>5. Sets expectedOutput to either [1,0,0] or [0,x,y] where x + y = 1 depending on if Adjusted-learning-rate is positive or negative.
     * <br>6. Calls {@link NN#learn} on these parameters to update the Policy using the advantage value
     */
    private static void updatePolicyOnEpisode() {
        List<Pair<Pair<State, Action>, Double>> data = runEpisode();
        int x = 0;

        for (int i = 0; i < data.size(); i++) {
            State s = data.get(i).first().first();
            double targetValue = 0;
            for (int j = data.size() - 1; j >= i; j--)
                targetValue = discountRate * targetValue + data.get(j).second();
            double advantageValue = targetValue - valueFunction.calculateOutput(new double[]{s.x,s.y})[0];

            NN.learn(valueFunction,valueFunctionLearningRate,new double[][]{{s.x, s.y}},new double[][]{{targetValue}});

            double[] output = Policy.calculateOutput(new double[]{s.x, s.y});
            double adjustedGradientMultiplier = learningRate * Math.pow(discountRate, i) * advantageValue / (output[data.get(i).first().second().ordinal()] + 1e-3);
            if (adjustedGradientMultiplier >= 0) {
                output = new double[3];
                output[data.get(i).first().second().ordinal()] = 1;
            } else {
                output[data.get(i).first().second().ordinal()] = 0;
                double sum = output[0] + output[1] + output[2];
                output[0] /= sum;
                output[1] /= sum;
                output[2] /= sum;
            }
            NN.learn(Policy, Math.abs(adjustedGradientMultiplier), new double[][]{{s.x, s.y}}, new double[][]{output});
        }
    }

    /** Runs an episode of the REINFORCE algorithm, returns state-action-reward tuple list */
    private static List<Pair<Pair<State, Action>, Double>> runEpisode() {
        ArrayList<Pair<Pair<State, Action>, Double>> data = new ArrayList<>();

        State currentState = new State(0.4 + Math.random() * 0.2, 0);
        while (currentState.y <= 1) {
            Action action = getPolicyDecision(currentState);

            double x = currentState.x, y = currentState.y + Math.random() * 0.02 + 0.04;
            if (action == Action.UPLEFT)
                x -= Math.random() * 0.02 + 0.04;
            else if (action == Action.UPRIGHT)
                x += Math.random() * 0.02 + 0.04;

            x -= stateToWindValue.apply(currentState);
            x = Math.clamp(x, 0, 1);
            State nextState = new State(x, y);

            Pair<State, Action> stateActionPair = new Pair<>(currentState, action);
            double reward = stateTransitionToReward.apply(stateActionPair, nextState);
            data.add(new Pair<>(stateActionPair, reward));

            currentState = nextState;
        }

        return data;
    }

    private static Action getPolicyDecision(State s) {
        double[] actionProbabilities = Policy.calculateOutput(new double[]{s.x, s.y});

        double random = Math.random();
        for (int i = 0; i < actionProbabilities.length; i++)
            if ((random -= actionProbabilities[i]) <= 0)
                return Action.values()[i];

        throw new RuntimeException("Unexpected error in Policy output" + Arrays.toString(actionProbabilities));
    }

    private record State(double x, double y) {
    }

    /** Either only moves up 0.2 ~ 0.4 units, or moves left or right 0.2 ~ 0.4 units as well. */
    private enum Action {UPLEFT, UP, UPRIGHT}
}
