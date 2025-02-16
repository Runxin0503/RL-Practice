import java.util.*;
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

    private static Queue<Integer> cards;

    /** {@code qπ(s,a)}: Given state s and action a, returns the expected future reward of the state action pair assuming {@link #policy} is followed afterward */
    private static HashMap<Pair<State,Action>, Double> actionValueFunction;

    /** {@code π(s,a)}: Given a State s and (valid) Action a, returns the probability of taking Action a in State s */
    private static HashMap<Pair<State, Action>,Double> policy;

    /** {@code r}: In this instance, the stateToReward is associated purely to a state, however in most
     * cases it is associated with a state-action pair (s,a) or a state transition (s,a,s') */
    private static final BiFunction<Pair<State,Action>,State,Double> stateTransitionToReward = (stateActionPair, nextState) -> {
        //todo implement card counting and card draws
        throw new UnsupportedOperationException("Not supported yet.");
    };

    /** {@code α}, otherwise known as the step size, controls the learning rate of the policy and how fast it converges */
    private static final double alpha = 2e-4;

    /** The Discount Rate Parameter of the Blackjack problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    /** The probability of the policy choosing an action randomly and uniformly from the possible action values */
    private static final double epsilon = 1e-2;

    static {
        actionValueFunction = new HashMap<>();
        policy = new HashMap<>();

        resetDeck();

        for(int dealerCard = 0;dealerCard <=10;dealerCard++)
            for(int currentSum = 12;currentSum <=21;currentSum++)
                for(boolean usableAce : new boolean[]{true,false})
                    for(Action action : Action.values()) {
                        Pair<State,Action> pair = new Pair<>(new State(dealerCard, currentSum, usableAce), action);
                        actionValueFunction.put(pair, 0.0); //initialize actionValueFunction with all 0s
                        policy.put(pair,0.5); //initialize policy with equally likely chance of choosing either possible actions
                    }

    }

    public static void main(String[] args){
        //todo do smth here
    }

    /** Gets the data using the current policy and uses it to update the current action-value function approximation */
    private static void updateAgentWithData(){
        ArrayList<Pair<Pair<State,Action>,Double>> data = getDataWithPolicy();
        for(int t=0;t<data.size();t++){
            Pair<Pair<State,Action>,Double> stateActionReward = data.get(t);
            double gt = 0;
            for(int i=data.size()-1;i>=t+1;i--){
                gt *= discountRate;
                gt += data.get(i).second();
            }

            double currentValuation = actionValueFunction.get(stateActionReward.first());
            actionValueFunction.replace(stateActionReward.first(),currentValuation + alpha * (gt - currentValuation));
        }

        updatePolicyWithActionValueFunction();
    }

    private static void updatePolicyWithActionValueFunction(){
        for(int dealerCard = 0;dealerCard <=10;dealerCard++)
            for(int currentSum = 12;currentSum <=21;currentSum++)
                for(boolean usableAce : new boolean[]{true,false}) {
                    State s = new State(dealerCard, currentSum, usableAce);
                    Pair<State,Action> HIT = new Pair<>(s,Action.HIT), STICK = new Pair<>(s,Action.STICK);
                    boolean hitBestAction = actionValueFunction.get(HIT) > actionValueFunction.get(STICK);
                    policy.replace(hitBestAction ? HIT : STICK,1 - epsilon + (epsilon) / 2);
                    policy.replace(hitBestAction ? STICK : HIT,epsilon / 2);
                }
    }

    /** Uses {@link #policy} to obtain some data (state-action pairs associated with rewards) */
    private static ArrayList<Pair<Pair<State,Action>,Double>> getDataWithPolicy(){
        ArrayList<Pair<Pair<State,Action>,Double>> data = new ArrayList<>();
        //todo implement
        throw new UnsupportedOperationException();
    }


    private static void resetDeck(){
        ArrayList<Integer> deck = new ArrayList<>();
        //initialize cards with appropriate numbers
        for(int i=0;i<10;i++)
            for(int j=0;j<4;j++)
                deck.add(i);

        for(int i=0;i<12;i++) deck.add(10);
        Collections.shuffle(deck);
        cards = new LinkedList<>(deck);
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
