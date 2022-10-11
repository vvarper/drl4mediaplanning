package markovdecisionproblem;

import org.deeplearning4j.rl4j.space.DiscreteSpace;

public class BinaryInvestmentActionSpace extends DiscreteSpace {


    public BinaryInvestmentActionSpace(int nrTps) {
        super((int) Math.pow(2, nrTps) - 1);
    }

    @Override
    public Integer noOp() {
        return -1;
    }
}
