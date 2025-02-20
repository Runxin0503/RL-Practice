public class LinAlg {
    public static double dotProduct(double[] first,double[] second){
        double result = 0;
        for (int i = 0; i < first.length; i++) result += first[i] * second[i];
        return result;
    }

    public static double[] scale(double constant,double[] array){
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) result[i] = array[i] * constant;
        return result;
    }

    public static double[] add(double[] first,double[] second){
        assert first.length == second.length;
        double[] result = new double[first.length];

        for(int i=0;i<first.length;i++)
            result[i] = first[i] + second[i];
        return result;
    }
}
