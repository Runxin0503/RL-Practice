/** A pair of state and actions (Helper Class) */
public record Pair<T, R>(T first, R second) {

    @Override
    public int hashCode() {
        return Integer.hashCode(first.hashCode()) ^ second.hashCode();
    }

    @Override
    public String toString() {
        return "(" + first + " | " + second + ")";
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Pair && ((Pair<?, ?>) obj).second.equals(second) && ((Pair<?, ?>) obj).first.equals(first);
    }
}