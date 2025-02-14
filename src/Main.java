import java.util.HashMap;

/**
 * A state is represented as an int s where the gambler has (s) amount of dollars<br>
 * An Action is represented as an int a <= min(s,100-s) where the gambler will bet (a) amount of dollars<br>
 * A Reward is represented as a double and is associated purely to a state s
 */
public class Main {
    /** {@code vπ(s)}: Given a state s, returns the discounted stateToReward of that state assuming {@link #policy} is followed */
    private static final HashMap<Integer,Double> stateValueFunction;

    /** {@code qπ(s,a)}: Given a state s and an action a, returns the discounted stateToReward of the next states assuming {@link #policy} is followed */
    private static final HashMap<intPairs,Double> actionValueFunction;

    /** {@code π(s,a)}: Given a state s and an action a, returns the probability that this action is taken in state s */
    private static final HashMap<intPairs,Double> policy;

    /** {@code r}: In this instance, the stateToReward is associated purely to a state, however in most
     * cases it is associated with a state-action pair (s,a) or a state transition (s,a,s') */
    private static final HashMap<Integer,Double> stateToReward;

    /** The probability that the imaginary coin flip lands on head and the gambler wins his gambled money */
    private static final double probWinGamble = 0.4;
    private static final double discountRate = 1;

    static {
        stateValueFunction = new HashMap<>();
        policy = new HashMap<>();
        actionValueFunction = new HashMap<>();
        stateToReward = new HashMap<>();

        stateValueFunction.put(0,0.0);
        for(int state = 1; state <100; state++){
            stateValueFunction.put(state,0.0);
            for(int action = 1; action <Math.min(state,100-state); action++) {
                actionValueFunction.put(new intPairs(state, action), 0.0);
                policy.put(new intPairs(state, action),1.0/ state);
            }
        }
        stateValueFunction.put(100,0.0);
    }

    public static void main(String[] args) {

    }

    public static void updateStateValueFunction(){
        //todo implement reiterating and reassigning the state-value function according to the bellman's equation

    }

    public static double updateStateValue(int state){
        double newStateValue = 0;
        for(int action = 1; action <= Math.min(state,100-state); action++){
            //the gambler bets (action) amount of money, possible states are either win or lose that money
            double nextStateValueSum = 0;
            for (int nextState : new int[]{state+action,state-action}){
                //stateToReward is defined by the nextState
                double reward = stateToReward.get(nextState);
                //p(s',r|s,a) = (winning ? probWinGamble : 1 - probWinGamble)
                nextStateValueSum += (nextState > state ? probWinGamble : 1 - probWinGamble) * (reward + discountRate * stateValueFunction.get(nextState));
            }

            newStateValue += policy.get(new intPairs(state,action)) * nextStateValueSum;
        }

        return newStateValue;
    }
}