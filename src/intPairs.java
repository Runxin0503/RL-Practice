
/** A Memory Efficient wrapper for pairs of ints */
public class intPairs {
    private final long combined;

    public intPairs(int num1, int num2) {
        this.combined = ((long) num1 << 32) | (num2 & 0xFFFFFFFFL);
    }

    public int first() {
        return (int) (combined >> 32);
    }

    public int second() {
        return (int) combined;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(combined);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof intPairs && combined == ((intPairs) obj).combined;
    }
}