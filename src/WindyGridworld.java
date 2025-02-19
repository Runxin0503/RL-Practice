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

    /** The Discount Rate Parameter of the Blackjack problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    /** The probability of a policy choosing an action randomly and uniformly from the possible action values */
    private static final double epsilon = 0.1;

    /** The number of steps to take in a Trajectory (episode) before updating the current policy */
    private static final int n = 1;

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
            for (int column = 0; column <= columns; column++) {
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
        //todo implement
    }

    /** Runs one episode of SARSA, calling {@link #updateSARSA(State, Action, double, State, Action)} every time a tuple (s,a,r,s',a') is accumulated. */
    private static void runSARSAEpisode() {
        State currentState = startingState, newState = null;
        Action currentAction = selectActionFromPolicy(currentState, SARSAPolicy);
        while (!currentState.equals(goalState)) {
            switch (currentAction) {
                case LEFT -> newState = new State(currentState.row, currentState.column - 1);
                case RIGHT -> newState = new State(currentState.row, currentState.column + 1);
                case UP -> {
                    if (windLevel[currentState.column] != 0) {
                        int randomSample = new int[]{-1, 0, 1}[(int) (Math.random() * 3)];
                        int newRow = Math.clamp(currentState.row + randomSample - 1, 0, rows);
                        newState = new State(newRow, currentState.column);
                    } else {
                        newState = new State(currentState.row - 1, currentState.column);
                    }
                }
                case DOWN -> {
                    if (windLevel[currentState.column] != 0) {
                        int randomSample = new int[]{-1, 0, 1}[(int) (Math.random() * 3)];
                        int newRow = Math.clamp(currentState.row + randomSample + 1, 0, rows);
                        newState = new State(newRow, currentState.column);
                    } else {
                        newState = new State(currentState.row + 1, currentState.column);
                    }
                }
                case null, default -> throw new RuntimeException("Unexpected error in returned action from policy");
            }

            Action nextAction = selectActionFromPolicy(newState, SARSAPolicy);
            updateSARSA(currentState,currentAction,reward,newState,nextAction);
            currentState = newState;
            currentAction = nextAction;
        }
    }

    private static void updateSARSA(State s,Action a,double r,State nextState,Action nextAction) {
        //todo implement
    }

    private static Action selectActionFromPolicy(State s, HashMap<Pair<State, Action>, Double> policy) {
        double random = Math.random();
        for (Action a : getValidActions(s)) {
            random -= policy.get(new Pair<>(s, a));
            if (random < 0) return a;
        }
        throw new RuntimeException("Unexpected error in sampling from policy");
    }


    private static List<Action> getValidActions(State state) {
        ArrayList<Action> actions = new ArrayList<>(4);
        if (state.row != 0) actions.add(Action.UP);
        if (state.column != 0) actions.add(Action.LEFT);
        if (state.row != rows) actions.add(Action.DOWN);
        if (state.column != columns) actions.add(Action.RIGHT);
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