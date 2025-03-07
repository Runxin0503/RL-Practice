import Network.Activation;
import Network.Cost;
import Network.NN;
import Utils.LinAlg;
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

    private static final BiFunction<Pair<State, Action>, State, Double> stateTransitionToReward = (stateActionPair, nextState) -> Math.sin(nextState.x * 2 * Math.PI);

    private static final Function<State, Double> stateToWindValue = state -> Math.cos(state.x * 2 * Math.PI) * 0.02;

    private static final NN Policy;

    private static final double discountRate = 1;

    private static final double learningRate = 1e-5;

    static {
        Policy = new NN.NetworkBuilder().setInputNum(2).setHiddenAF(Activation.ReLU)
                .setOutputAF(Activation.softmax).setCostFunction(Cost.diffSquared)
                .addDenseLayer(120).addDenseLayer(60).addDenseLayer(20)
                .addDenseLayer(3).build();
    }

    public static void main(String[] args) {
        for (int i = 0; i < 50_000; i++) {
            updatePolicyOnEpisode();
            System.out.println(i);
        }
        runEpisode().forEach(System.out::println);
        updatePolicyOnEpisode();
    }
    
    private static void updatePolicyOnEpisode() {
        List<Pair<Pair<State,Action>,Double>> data = runEpisode();
        int x = 0;

        for (int i = 0; i < data.size(); i++) {
            double targetValue = 0;
            for (int j = data.size() - 1; j >= i; j--) {
                targetValue = discountRate * targetValue + data.get(j).second();
            }

            State s = data.get(i).first().first();
            double[] output = Policy.calculateOutput(new double[]{s.x, s.y});
            double adjustedGradientMultiplier = learningRate * Math.pow(discountRate,i) * targetValue / output[data.get(i).first().second().ordinal()];
            output[data.get(i).first().second().ordinal()] = adjustedGradientMultiplier >= 0 ? 1 : 0;
            NN.learn(Policy,Math.abs(adjustedGradientMultiplier),new double[][]{{s.x,s.y}},new double[][]{output});
        }
    }

    /** Runs an episode of naive policy gradient algorithm, returns state-action-reward tuple list */
    private static List<Pair<Pair<State, Action>, Double>> runEpisode() {
        ArrayList<Pair<Pair<State, Action>, Double>> data = new ArrayList<>();

        State currentState = new State(0.4 + Math.random() * 0.2, 0);
        while (currentState.y <= 1) {
            Action action = getPolicyDecision(currentState);

            double x = currentState.x, y = currentState.y + Math.random() * 0.02 + 0.04;
            if (action == Action.UPLEFT)
                x -= Math.random() * 0.02 + 0.04;
            else if(action == Action.UPRIGHT)
                x += Math.random() * 0.02 + 0.04;

            x -= stateToWindValue.apply(currentState);
            x = Math.clamp(x,0,1);
            State nextState = new State(x, y);

            Pair<State,Action> stateActionPair = new Pair<>(currentState, action);
            double reward = stateTransitionToReward.apply(stateActionPair,nextState);
            data.add(new Pair<>(stateActionPair, reward));

            currentState = nextState;
        }

        return data;
    }

    private static Action getPolicyDecision(State s) {
        double[] actionProbabilities = Policy.calculateOutput(new double[]{s.x,s.y});

        double random = Math.random();
        for(int i=0;i<actionProbabilities.length;i++)
            if((random -= actionProbabilities[i]) <= 0)
                return Action.values()[i];

        throw new RuntimeException("Unexpected error in Policy output" + Arrays.toString(actionProbabilities));
    }

    private record State(double x, double y) {}

    /** Either moves just up 0.2 ~ 0.4 units, or left or right 0.2 ~ 0.4 units as well. */
    private enum Action {UPLEFT, UP, UPRIGHT}
}
