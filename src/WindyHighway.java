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

    private static final BiFunction<Pair<State,Action>,State,Double> stateToReward = (stateActionPair, state) -> Math.sin(state.x * 2 * Math.PI);

    private static final Function<State,Double> stateToWindValue = state -> Math.cos(state.x * 2 * Math.PI) * 0.02;



    /** Runs an episode of naive policy gradient algorithm, returns state-action-reward tuple list */
    private static List<Pair<Pair<State,Action>,Double>> runEpisode(){
        //todo implement
    }

    private static Action getPolicyDecision(State s) {
        //todo implement
        throw new UnsupportedOperationException();
    }

    private record State(double x,double y){}
    private enum Action{UPLEFT,UP,UPRIGHT}
}
