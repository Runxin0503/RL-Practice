import java.util.HashMap;
import java.util.function.BiFunction;

/**
 * A state is represented as an int s where the gambler has (s) amount of dollars<br>
 * An Action is represented as an int a <= min(s,100-s) where the gambler will bet (a) amount of dollars<br>
 * A Reward is represented as a double and is associated purely to a state s
 */
public class Main {
    /** {@code vπ(s)}: Given a state s, returns the discounted stateToReward of that state assuming {@link #policy} is followed */
    private static HashMap<Integer, Double> stateValueFunction;

    /** {@code π(s)}: Given a state s, returns the greedy deterministic action to take for that state */
    private static HashMap<Integer, Integer> policy;

    /** {@code r}: In this instance, the stateToReward is associated purely to a state, however in most
     * cases it is associated with a state-action pair (s,a) or a state transition (s,a,s') */
    private static final BiFunction<intPairs,Integer,Double> stateTransitionToReward = (stateActionPair, nextState) -> (nextState == 100 ? 1.0 : 0.0);

    /** The probability that the imaginary coin flip lands on head and the gambler wins his gambled money */
    private static final double probWinGamble = 0.4;

    /** The Discount Rate Parameter of Gambler's Problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    /** epsilon Parameter is used in tie-breaker situations when finding a
     * specific deterministic policy out of many equally optimal-performing policies */
    private static final double epsilon = 1e-5;

    static {
        stateValueFunction = new HashMap<>();
        policy = new HashMap<>();

        for (int state = 0; state <= 100; state++)
            stateValueFunction.put(state, 0.0);
        improvePolicy();
    }

    public static void main(String[] args) {
//        improvePolicy();
//        updateStateValueFunction();
//        System.out.println(stateValueFunction);
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1; j++)
                updateStateValueFunction();
            improvePolicy();
            System.out.println(i);
        }
        System.out.println("stateValueFunction: \n" + stateValueFunction);
        System.out.println("\n\npolicy: ");
        for(int state = 1; state < 100; state++){
            System.out.println(state+","+policy.get(state));
        }
    }

    /** Creates a new policy that is deterministic and greedy with respect to the action value function */
    public static void improvePolicy() {
        HashMap<Integer,Integer> newPolicy = new HashMap<>();
        for (int state = 1; state < 100; state++) {
            newPolicy.put(state,getBestActionFromState(state));
        }
        policy = newPolicy;
    }

    public static int getBestActionFromState(int state) {
        int maxAction = 0;
        double maxActionValue = Double.NEGATIVE_INFINITY;
        for (int action = 1; action <= Math.min(state, 100 - state); action++) {
            double actionValue = 0;
            for (int nextState : new int[]{state + action, state - action}) {
                //stateToReward is defined by the nextState
                double reward = stateTransitionToReward.apply(new intPairs(state,action),nextState);
                //p(s',r|s,a) = (winning ? probWinGamble : 1 - probWinGamble)
                actionValue += (nextState > state ? probWinGamble : 1 - probWinGamble) * (reward + discountRate * stateValueFunction.getOrDefault(nextState,0.0));
            }

            if (actionValue > maxActionValue + epsilon) {
                maxAction = action;
                maxActionValue = actionValue;
            }
        }
        if(maxAction == 0 && maxActionValue == Double.NEGATIVE_INFINITY) throw new RuntimeException();

        return maxAction;
    }

    /** Reiterates and reassign the state-value function according to the bellman's equation and the current policy */
    public static void updateStateValueFunction() {
        HashMap<Integer, Double> newStateValueFunction = new HashMap<>();
        for (int i = 1; i < 100; i++) {
            newStateValueFunction.put(i, updateStateValue(i));
        }
        newStateValueFunction.put(0,0.0);
        newStateValueFunction.put(100,0.0);
        stateValueFunction = newStateValueFunction;
    }

    /** Returns the updated state-value function's output at this specific state.<br>
     * Also updates the output of action-value function of all outgoing actions from this state */
    public static double updateStateValue(int state) {
        int action = policy.get(state);
        //the gambler bets (action) amount of money, possible states are either win or lose that money
        double actionValue = 0;
        for (int nextState : new int[]{state + action, state - action}) {
            //stateToReward is defined by the nextState
            double reward = stateTransitionToReward.apply(new intPairs(state, action), nextState);
            //p(s',r|s,a) = (winning ? probWinGamble : 1 - probWinGamble)
            actionValue += (nextState > state ? probWinGamble : 1 - probWinGamble) * (reward + discountRate * stateValueFunction.get(nextState));
        }

        return actionValue;
    }
}