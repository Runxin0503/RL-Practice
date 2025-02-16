/** A pair of state and actions */
public record stateActionPairs<T, R>(T state, R action) {

    @Override
    public int hashCode() {
        return Integer.hashCode(state.hashCode()) ^ action.hashCode();
    }

    @Override
    public String toString() {
        return "(" + state + " | " + action + ")";
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof stateActionPairs && ((stateActionPairs<?, ?>) obj).action.equals(action) && ((stateActionPairs<?, ?>) obj).state.equals(state);
    }
}