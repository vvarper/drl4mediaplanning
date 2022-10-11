package markovdecisionproblem;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

public class ABMEnv implements MDP<Observation, Integer, DiscreteSpace> {

    private final ObservationSpace<Observation> observationSpace;
    private final BinaryInvestmentActionSpace actionSpace;
    private final int initialSeed;
    private final ABMBuilder modelBuilder;
    private final double[] historicalAwareness;
    private ABM model;
    private int seed;

    public ABMEnv(ABMBuilder modelBuilder, double[] historicalAwareness,
                  int initialSeed) {

        int observationSize = 3 + modelBuilder.getNumberOnlineTouchpoints();
        observationSpace = new ArrayObservationSpace<>(new int[]{1, observationSize});
        actionSpace = new BinaryInvestmentActionSpace(
                modelBuilder.getNumberOnlineTouchpoints()
        );

        this.modelBuilder = modelBuilder;
        this.historicalAwareness = historicalAwareness;
        this.initialSeed = initialSeed;
        seed = initialSeed;

    }

    @Override
    public Observation reset() {
        close();
        return model.getObservation();
    }

    @Override
    public void close() {
        model = modelBuilder.buildABM(seed);
        seed = (seed + 1) % 30;
    }

    @Override
    public ObservationSpace<Observation> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public BinaryInvestmentActionSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public StepReply<Observation> step(Integer action) {
        double reward = model.runWindowStepsAction(action + 1, historicalAwareness);
        return new StepReply<>(model.getObservation(), reward, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return model.isSimulationFinished();
    }

    @Override
    public MDP<Observation, Integer, DiscreteSpace> newInstance() {
        return new ABMEnv(modelBuilder, historicalAwareness, initialSeed);
    }
}
