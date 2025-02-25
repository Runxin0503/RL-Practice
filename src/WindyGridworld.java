import Utils.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * A Probabilistic Grid world simulation that compares the performances of SARSA, Expected SARSA, and Q-Learning.
 * <br>All algorithms have {@code alpha = 0.25}, {@code epsilon = 0.1}, and are adjusted from {@code 1-step TD Control}.
 * <br> All rewards from state transitions are -1. This incentivizes the algorithms to get as quickly to the goal point as possible to lose the least number of rewards
 */
public class WindyGridworld {

    /** The dimensions of the windy gridworld */
    private static final int rows = 7, columns = 10;

    /** The coordinate of the start and goal point */
    private static final State startingState = new State(3, 0), goalState = new State(3, 7);

    /** The wind level of each column in the windy gridworld. When moving out of a wind column,
     * the player gets pushed up x grids, where x = wind level + uniformly chosen {-1,0,1} */
    private static final int[] windLevel = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 1, 0};

    /** {@code qπ(s,a)}: Given state s and action a, returns the expected future reward of the state action pair assuming {@link #SARSAPolicy} is followed afterward */
    private static final HashMap<Pair<State, Action>, Double> SARSAActionValueFunction;

    /** {@code qπ(s,a)}: Given state s and action a, returns the expected future reward of the state action pair assuming {@link #expectedSARSAPolicy} is followed afterward */
    private static final HashMap<Pair<State, Action>, Double> expectedSARSAActionValueFunction;

    /** {@code qπ(s,a)}: Given state s and action a, returns the expected future reward of the state action pair assuming {@link #qLearningPolicy} is followed afterward */
    private static final HashMap<Pair<State, Action>, Double> qLearningActionValueFunction;

    /** {@code π(s,a)}: Given State s and (valid) Action a, returns the probability of taking Action a in State s */
    private static final HashMap<Pair<State, Action>, Double> SARSAPolicy;

    /** {@code π(s,a)}: Given State s and (valid) Action a, returns the probability of taking Action a in State s */
    private static final HashMap<Pair<State, Action>, Double> expectedSARSAPolicy;

    /** {@code π(s,a)}: Given State s and (valid) Action a, returns the probability of taking Action a in State s */
    private static final HashMap<Pair<State, Action>, Double> qLearningPolicy;

    /** {@code α}, otherwise known as the step size, controls the learning rate of the policy and how fast it converges */
    private static final double alpha = 0.25;

    /** The Discount Rate Parameter of the WindyGridworld problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    /** The probability of a policy choosing an action randomly and uniformly from the possible action values */
    private static final double epsilon = 0.1;

    /** A Constant Reward given to the algorithms upon every action they take. This encourages the algorithms to
     * minimize the total number of actions taken and thus the total number of paths traversed until the goal point */
    private static final double reward = -1;

    static {
        SARSAPolicy = new HashMap<>();
        expectedSARSAPolicy = new HashMap<>();
        qLearningPolicy = new HashMap<>();
        SARSAActionValueFunction = new HashMap<>();
        expectedSARSAActionValueFunction = new HashMap<>();
        qLearningActionValueFunction = new HashMap<>();

        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++) {
                State s = new State(row, column);
                List<Action> validActions = getValidActions(s);
                for (Action action : validActions) {
                    Pair<State, Action> pair = new Pair<>(s, action);
                    SARSAActionValueFunction.put(pair, 0.0); //initialize actionValueFunction with all 0s
                    expectedSARSAActionValueFunction.put(pair, 0.0);
                    qLearningActionValueFunction.put(pair, 0.0);
                    SARSAPolicy.put(pair, 1.0 / validActions.size()); //initialize policy with equally likely chance of choosing either possible actions
                    expectedSARSAPolicy.put(pair, 1.0 / validActions.size());
                    qLearningPolicy.put(pair, 1.0 / validActions.size());
                }
            }
    }

    public static void main(String[] args) {
        Thread[] threadGroup = new Thread[3];
        for (int i = 0; i < 100_000; i++) {
            threadGroup[0] = new Thread(WindyGridworld::runSARSAEpisode);
            threadGroup[1] = new Thread(() -> run4TupleAlgorithmEpisode(expectedSARSAPolicy, expectedSARSAActionValueFunction));
            threadGroup[2] = new Thread(() -> run4TupleAlgorithmEpisode(qLearningPolicy, qLearningActionValueFunction));

            for (Thread t : threadGroup) t.start();
            for (Thread t : threadGroup)
                try {
                    t.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            System.out.println(i);
        }

        for (HashMap<?, ?> policy : new HashMap<?, ?>[]{SARSAPolicy, expectedSARSAPolicy, qLearningPolicy}) {
            if (policy == SARSAPolicy) System.out.print("SARSA Policy: ");
            else if (policy == expectedSARSAPolicy) System.out.print("Expected SARSA Policy: ");
            else if (policy == qLearningPolicy) System.out.print("Q Learning Policy: ");

            double avgReward = 0;
            for (int i = 0; i < 1_000_000; i++)
                avgReward += getRewardFromEpisode((HashMap<Pair<State, Action>, Double>) policy);
            avgReward /= 1_000_000;
            System.out.println(avgReward);

            for (int row = 0; row < rows; row++) {
                System.out.print("[");
                for (int column = 0; column < columns; column++) {
                    State s = new State(row, column);
                    Action bestAction = null;
                    for (Action a : getValidActions(s)) if ((Double) policy.get(new Pair<>(s, a)) > 0.5) bestAction = a;
                    System.out.print(switch (bestAction) {
                        case UP -> "^";
                        case DOWN -> "v";
                        case LEFT -> "<";
                        case RIGHT -> ">";
                        case null -> "?";
                    } + ",");
                }
                System.out.println("]");
            }
            System.out.println("\n\n");
        }
    }

    private static double getRewardFromEpisode(HashMap<Pair<State, Action>, Double> policy) {
        State currentState = startingState;
        Action currentAction;
        double totalReward = 0;

        while (!currentState.equals(goalState)) {
            currentAction = sampleActionFromPolicy(currentState, policy);
            switch (currentAction) {
                case LEFT -> currentState = new State(getRowWithWind(currentState, 0), currentState.column - 1);
                case RIGHT -> currentState = new State(getRowWithWind(currentState, 0), currentState.column + 1);
                case UP -> currentState = new State(getRowWithWind(currentState, -1), currentState.column);
                case DOWN -> currentState = new State(getRowWithWind(currentState, 1), currentState.column);
                case null, default -> throw new RuntimeException("Unexpected error in returned action from policy");
            }
            totalReward += reward;
        }

        return totalReward;
    }

    private static int getRowWithWind(State s, int delta) {
        if (windLevel[s.column] != 0) {
            int randomSample = new int[]{-1, 0, 1}[(int) (Math.random() * 3)];
            return Math.clamp(s.row + randomSample + delta, 0, rows - 1);
        }
        return s.row;
    }

    /** Runs one episode of SARSA, calling {@link #updateSARSA(State, Action, double, State, Action)} every time a tuple (s,a,r,s',a') is accumulated. */
    private static void runSARSAEpisode() {
        State currentState = startingState, newState;
        Action currentAction = sampleActionFromPolicy(currentState, SARSAPolicy);
        while (!currentState.equals(goalState)) {
            switch (currentAction) {
                case LEFT -> newState = new State(getRowWithWind(currentState, 0), currentState.column - 1);
                case RIGHT -> newState = new State(getRowWithWind(currentState, 0), currentState.column + 1);
                case UP -> newState = new State(getRowWithWind(currentState, -1), currentState.column);
                case DOWN -> newState = new State(getRowWithWind(currentState, 1), currentState.column);
                case null, default -> throw new RuntimeException("Unexpected error in returned action from policy");
            }

            Action nextAction = sampleActionFromPolicy(newState, SARSAPolicy);
            updateSARSA(currentState, currentAction, reward, newState, nextAction);
            currentState = newState;
            currentAction = nextAction;
        }
    }

    /** Follows the update rule of Q(s,a) <- r + discountRate * Q(s',a') to update {@link #SARSAActionValueFunction}.
     * Then, scans through the values of SARSA's policy π(s,x) for all valid action x in state s and epsilon-greedily updates them */
    private static void updateSARSA(State s, Action a, double r, State nextState, Action nextAction) {

        double targetValue = r + (nextState.equals(goalState) ? 0 : discountRate * SARSAActionValueFunction.get(new Pair<>(nextState, nextAction)));

        Pair<State, Action> stateActionPair = new Pair<>(s, a);
        double currentValue = SARSAActionValueFunction.get(stateActionPair);
        SARSAActionValueFunction.replace(stateActionPair, currentValue + alpha * (targetValue - currentValue));

        Action bestAction = getBestActionFromActionValueFunction(s, SARSAActionValueFunction);
        List<Action> validActions = getValidActions(s);
        for (Action action : validActions) {
            SARSAPolicy.replace(new Pair<>(s, action), (action == bestAction ? 1 - epsilon : 0) + epsilon / validActions.size());
        }
    }


    /** Runs one episode of either Q-Learning or expectedSARSA, which doesn't require the 5-tuple (s,a,r,s',a') and instead only requires the 4-tuple (s,a,r,s').
     * Calls {@link #updateSARSA} every time the 4-tuple is accumulated. */
    private static void run4TupleAlgorithmEpisode(HashMap<Pair<State, Action>, Double> policy, HashMap<Pair<State, Action>, Double> actionValueFunction) {
        State currentState = startingState, newState;
        Action currentAction = sampleActionFromPolicy(currentState, policy);
        while (!currentState.equals(goalState)) {
            switch (currentAction) {
                case LEFT -> newState = new State(getRowWithWind(currentState, 0), currentState.column - 1);
                case RIGHT -> newState = new State(getRowWithWind(currentState, 0), currentState.column + 1);
                case UP -> newState = new State(getRowWithWind(currentState, -1), currentState.column);
                case DOWN -> newState = new State(getRowWithWind(currentState, 1), currentState.column);
                case null, default -> throw new RuntimeException("Unexpected error in returned action from policy");
            }

            update4TupleAlgorithm(currentState, currentAction, reward, newState, policy, actionValueFunction);
            currentState = newState;
            currentAction = sampleActionFromPolicy(currentState, policy);
        }
    }

    /** Follows the update rule of either Q-Learning or Expected SARSA to update the given {@code actionValueFunction}.
     * Then, scans through the values of the given {@code policy} π(s,x) for all valid action x in state s and epsilon-greedily updates them */
    private static void update4TupleAlgorithm(State s, Action a, double r, State nextState, HashMap<Pair<State, Action>, Double> policy, HashMap<Pair<State, Action>, Double> actionValueFunction) {

        double targetValue = 0;
        if (actionValueFunction == qLearningActionValueFunction) {
            Action bestAction = getBestActionFromActionValueFunction(nextState, actionValueFunction);
            targetValue = actionValueFunction.get(new Pair<>(nextState, bestAction));
        } else {
            for (Action validAction : getValidActions(nextState)) {
                Pair<State, Action> nextStateActionPair = new Pair<>(nextState, validAction);
                targetValue += actionValueFunction.get(nextStateActionPair) * policy.get(nextStateActionPair);
            }
        }
        targetValue = r + (nextState.equals(goalState) ? 0 : discountRate * targetValue);

        Pair<State, Action> stateActionPair = new Pair<>(s, a);
        double currentValue = actionValueFunction.get(stateActionPair);
        actionValueFunction.replace(stateActionPair, currentValue + alpha * (targetValue - currentValue));

        Action bestAction = getBestActionFromActionValueFunction(s, actionValueFunction);
        List<Action> validActions = getValidActions(s);
        for (Action action : validActions) {
            policy.replace(new Pair<>(s, action), (action == bestAction ? 1 - epsilon : 0) + epsilon / validActions.size());
        }
    }

    /** Gets the highest valued action a at State s according to the input {@code actionValueFunction} */
    private static Action getBestActionFromActionValueFunction(State s, HashMap<Pair<State, Action>, Double> actionValueFunction) {
        Action bestAction = null;
        double bestActionValue = -Double.MAX_VALUE;
        for (Action a : getValidActions(s)) {
            double actionValue = actionValueFunction.get(new Pair<>(s, a));
            if (actionValue > bestActionValue) {
                bestAction = a;
                bestActionValue = actionValue;
            }
        }

        assert bestAction != null;
        return bestAction;
    }

    /** Randomly samples an action from the distribution of possible actions provided by the policy */
    private static Action sampleActionFromPolicy(State s, HashMap<Pair<State, Action>, Double> policy) {
        double random = Math.random();
        for (Action a : getValidActions(s)) {
            random -= policy.get(new Pair<>(s, a));
            if (random < 0) return a;
        }
        throw new RuntimeException("Unexpected error in sampling from policy");
    }


    private static List<Action> getValidActions(State state) {
        ArrayList<Action> actions = new ArrayList<>(4);
        if (state.row > 0) actions.add(Action.UP);
        if (state.column > 0) actions.add(Action.LEFT);
        if (state.row < rows - 1) actions.add(Action.DOWN);
        if (state.column < columns - 1) actions.add(Action.RIGHT);
        return actions;
    }

    private record State(int row, int column) {
        @Override
        public boolean equals(Object obj) {
            return (obj instanceof State) && ((State) obj).row == row && ((State) obj).column == column;
        }
    }

    private enum Action {UP, DOWN, LEFT, RIGHT}
}