import java.util.HashMap;
import java.util.function.BiFunction;

/**
 * A Discrete Model-Based Markov-Decision-Process (MDP) where the problem
 * statement is defined as a blackjack player trying to win blackjack by either
 * hitting or sticking. This code uses Constant-alpha Monte Carlos to solve and
 * find (one of) the optimal strategy to do so.<br>
 * A State is represented as the class {@link State}<br>
 * An Action is represented as either hit or stick in {@link Action}, where choosing to stick
 * instantly transitions the player into a terminal State (win, lose, draw).<br>
 * A Reward is associated to a State-transition (s,a,s') where if s' is a terminal State then a reward is given.
 * {@code Reward r = win: 1, draw: 0, lose: -1}.
 */
public class blackjack {

    /** {@code qπ(s,a)}: Given state s and action a, returns the expected future reward of the state action pair assuming {@link #policy} is followed afterward */
    private static HashMap<stateActionPairs<State,Action>, Double> actionValueFunction;

    /** {@code π(s,a)}: Given a State s and (valid) Action a, returns the probability of taking Action a in State s */
    private static HashMap<stateActionPairs<State, Action>,Double> policy;

    /** {@code r}: In this instance, the stateToReward is associated purely to a state, however in most
     * cases it is associated with a state-action pair (s,a) or a state transition (s,a,s') */
    private static final BiFunction<stateActionPairs<State,Action>,State,Double> stateTransitionToReward = (stateActionPair, nextState) -> {
        //todo implement card counting and card draws
        throw new UnsupportedOperationException("Not supported yet.");
    };

    /** {@code α}, otherwise known as the step size, controls the learning rate of the policy and how fast it converges */
    private static final double alpha = 2e-4;

    /** The probability of the policy choosing an action randomly and uniformly from the possible action values */
    private static final double epsilon = 1e-2;

    static {
        actionValueFunction = new HashMap<>();
        policy = new HashMap<>();

        //todo implement initialization
    }

    /**
     * @param dealerCard  The dealer's card from 1 to 10. If it's 1 then it's an ace, which means it can count as 1 or 11 in blackjack
     * @param currentSum  The current sum of the agent's deck, from 12 to 21. Any State where currentSum reaches above 21 is an instant terminal State
     * @param usableAce  If the agent's deck contains an ace or not  */
    private record State(int dealerCard, int currentSum, boolean usableAce) {

        @Override
            public boolean equals(Object obj) {
                return (obj instanceof State) && ((State) obj).dealerCard == dealerCard && ((State) obj).currentSum == currentSum && ((State) obj).usableAce == usableAce;
            }
        }
    private enum Action {
        HIT,STICK;
    }
}
