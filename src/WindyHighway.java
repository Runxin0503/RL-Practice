import java.util.ArrayList;
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

    private static final BiFunction<Pair<State, Action>, State, Double> stateTransitionToReward = (stateActionPair, state) -> Math.sin(state.x * 2 * Math.PI);

    private static final Function<State, Double> stateToWindValue = state -> Math.cos(state.x * 2 * Math.PI) * 0.02;

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
        //todo implement
        throw new UnsupportedOperationException();
    }

    private record State(double x, double y) {
    }

    /** Either moves just up 0.2 ~ 0.4 units, or left or right 0.2 ~ 0.4 units as well. */
    private enum Action {UPLEFT, UP, UPRIGHT}
}
