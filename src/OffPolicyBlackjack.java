import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * A Deterministic Model-Based On-Policy Markov-Decision-Process (MDP) where the problem
 * statement is defined as a blackjack player trying to win blackjack by either
 * hitting or sticking. This code uses Constant-alpha Monte Carlos to solve and
 * find (one of) the optimal strategy to do so.<br>
 * A State is represented as the class {@link State}<br>
 * An Action is represented as either hit or stick in {@link Action}, where choosing to stick
 * instantly transitions the player into a terminal State (win, lose, draw).<br>
 * A Reward is associated to a State-transition (s,a,s') where if s' is a terminal State then a reward is given.
 * {@code Reward r = win: 1, draw: 0, lose: -1}.
 */
public class OffPolicyBlackjack {

    /** {@code qπ(s,a)}: Given state s and action a, returns the expected future reward of the state action pair assuming {@link #policy} is followed afterward */
    private static HashMap<Pair<State, Action>, Double> actionValueFunction;

    /** {@code b(s,a)}: Given a State s and (valid) Action a, returns the probability of taking Action a in State s.
     * Serves as an exploration policy purely to collect data to train the target policy. */
    private static HashMap<Pair<State, Action>, Double> behaviorPolicy;

    /** {@code π(s)}: Given State s, returns the Action a the policy takes in State s.
     * Serves as the target policy that will approximate the optimal policy. */
    private static HashMap<State, Action> policy;

    /** {@code α}, otherwise known as the step size, controls the learning rate of the policy and how fast it converges */
    private static final double alpha = 2e-4;

    /** The Discount Rate Parameter of the Blackjack problem.
     * Set to 1 to not value future rewards any less than present ones */
    private static final double discountRate = 1;

    static {
        actionValueFunction = new HashMap<>();
        policy = new HashMap<>();
        behaviorPolicy = new HashMap<>();

        for (int dealerCard = 1; dealerCard <= 10; dealerCard++)
            for (int currentSum = 12; currentSum <= 21; currentSum++)
                for (boolean usableAce : new boolean[]{true, false})
                    for (Action action : Action.values()) {
                        Pair<State, Action> pair = new Pair<>(new State(dealerCard, currentSum, usableAce), action);
                        actionValueFunction.put(pair, 0.0); //initialize actionValueFunction with all 0s
                        policy.put(pair.first(), Math.random() < 0.5 ? Action.HIT : Action.STICK); //initialize policy with equally likely chance of choosing either possible actions
                        behaviorPolicy.put(pair, 0.5);
                    }
    }

    public static void main(String[] args) {
        for (int i = 0; i < 10_000_000; i++) {
            if (i % 10_000 == 0) System.out.println(i);
            updateAgentWithData();
        }

        System.out.println(actionValueFunction);

        for (boolean usableAce : new boolean[]{true, false})
            for (int dealerCard = 1; dealerCard <= 10; dealerCard++)
                for (int currentSum = 12; currentSum <= 21; currentSum++) {
                    State s = new State(dealerCard, currentSum, usableAce);
                    System.out.println(s + ": " + policy.get(s));
                }
    }

    /** Gets the data using the current policy and uses it to update the current action-value function approximation */
    private static void updateAgentWithData() {
        ArrayList<Pair<Pair<State, Action>, Double>> data = getDataWithPolicy();
        for (int t = 0; t < data.size(); t++) {
            Pair<State, Action> stateActionPair = data.get(t).first();
            double rho = 1;
            for (int i = t + 1; i < data.size(); i++) {
                Pair<State, Action> stateActionPair2 = data.get(i).first();
                if(policy.get(stateActionPair2.first()) == stateActionPair2.second()) rho /= behaviorPolicy.get(stateActionPair2);
                else {
                    rho = 0;
                    break;
                }
            }

            double gt = 0;
            for (int i = data.size() - 1; i >= t; i--)
                gt = discountRate * gt + data.get(i).second();
            gt *= rho;

            double currentValuation = actionValueFunction.get(stateActionPair);
            actionValueFunction.replace(stateActionPair, currentValuation + alpha * (gt - currentValuation));
        }

        updatePolicyWithActionValueFunction();
    }

    private static void updatePolicyWithActionValueFunction() {
        for (int dealerCard = 1; dealerCard <= 10; dealerCard++)
            for (int currentSum = 12; currentSum <= 21; currentSum++)
                for (boolean usableAce : new boolean[]{true, false}) {
                    State s = new State(dealerCard, currentSum, usableAce);
                    policy.replace(s, (actionValueFunction.get(new Pair<>(s, Action.HIT)) > actionValueFunction.get(new Pair<>(s, Action.STICK)) ? Action.HIT : Action.STICK));
                }
    }

    /** Uses {@link #policy} to obtain some data (state-action pairs associated with rewards) */
    private static ArrayList<Pair<Pair<State, Action>, Double>> getDataWithPolicy() {
        ArrayList<Pair<Pair<State, Action>, Double>> data = new ArrayList<>();

        ArrayList<Integer> agentCards = new ArrayList<>();
        agentCards.add(sampleDeck());
        agentCards.add(sampleDeck()); //get two cards

        int dealerFirstCard = sampleDeck();

        while (countCardNum(agentCards) < 12) agentCards.add(sampleDeck());
        State currentState = new State(dealerFirstCard, countCardNum(agentCards), checkUsableAce(agentCards));
        if (countCardNum(agentCards) == 21) {
            data.add(new Pair<>(new Pair<>(currentState, Action.STICK), 1.0));
            return data;
        }

        while (true) {
            Action policyAction = Math.random() < behaviorPolicy.get(new Pair<>(currentState, Action.HIT)) ? Action.HIT : Action.STICK;
            Pair<State, Action> stateActionPair = new Pair<>(currentState, policyAction);

            if (policyAction == Action.HIT) {

                //set the currentSum and usableAce to its correct value
                agentCards.add(sampleDeck());
                int currentSum = countCardNum(agentCards);
                if (currentSum > 21) {
                    data.add(new Pair<>(stateActionPair, -1.0));//lose
                    return data;
                } else if (currentSum == 21) {
                    data.add(new Pair<>(stateActionPair, 1.0));//win
                    return data;
                }

                data.add(new Pair<>(stateActionPair, 0.0));
                currentState = new State(dealerFirstCard, currentSum, checkUsableAce(agentCards));
            } else {
                ArrayList<Integer> dealerCards = new ArrayList<>();
                dealerCards.add(dealerFirstCard);
                dealerCards.add(sampleDeck());

                int dealerSum = countCardNum(dealerCards);
                while (dealerSum <= 16) {
                    int nextCard = sampleDeck();
                    dealerCards.add(nextCard);
                    dealerSum = countCardNum(dealerCards);
                }
                if (dealerSum == currentState.currentSum) //draw
                    data.add(new Pair<>(stateActionPair, 0.0));
                else if (dealerSum > 21 || dealerSum < currentState.currentSum) //win
                    data.add(new Pair<>(stateActionPair, 1.0));
                else //lose
                    data.add(new Pair<>(stateActionPair, -1.0));

                return data;
            }
        }
    }

    private static int sampleDeck() {
        return new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10}[(int) (Math.random() * 13)];
    }

    private static int countCardNum(List<Integer> cards) {
        boolean acePresent = false;
        int sum = 0;
        for (int card : cards) {
            sum += card;
            if (!acePresent && card == 1) acePresent = true;
        }

        if (acePresent && sum + 10 <= 21) return sum + 10;
        else return sum;
    }

    private static boolean checkUsableAce(List<Integer> cards) {
        boolean acePresent = false;
        int sum = 0;
        for (int card : cards) {
            sum += card;
            if (!acePresent && card == 1) acePresent = true;
        }

        return acePresent && sum + 10 <= 21;
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
        HIT, STICK;
    }
}
