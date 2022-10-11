package beans;

import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.nd4j.linalg.learning.config.Adam;

public class DQNConfig {

    private long seed;                  //Random seed (for reproducibility)
    private int maxEpochStep;           // Max step By epoch
    private int maxStep;                // Max step
    private int expRepMaxSize;          // Max size of experience replay
    private int batchSize;              // size of batches
    private int targetDqnUpdateFreq;    // target update (hard)
    private int updateStart;            // num step noop warmup
    private double rewardFactor;        // reward scaling
    private double minEpsilon;          // min epsilon
    private double gamma;               // gamma
    private double errorClamp;          // /td-error clipping
    private int epsilonNbStep;          // num step for eps greedy anneal
    private boolean doubleDQN;          // double DQN

    private double l2;
    private double learningRate;
    private int numHiddenNodes;         // num nodes per hidden layer
    private int numLayers;              // num hidden layers

    public long getSeed() {
        return seed;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public int getMaxEpochStep() {
        return maxEpochStep;
    }

    public void setMaxEpochStep(int maxEpochStep) {
        this.maxEpochStep = maxEpochStep;
    }

    public int getMaxStep() {
        return maxStep;
    }

    public void setMaxStep(int maxStep) {
        this.maxStep = maxStep;
    }

    public int getExpRepMaxSize() {
        return expRepMaxSize;
    }

    public void setExpRepMaxSize(int expRepMaxSize) {
        this.expRepMaxSize = expRepMaxSize;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getTargetDqnUpdateFreq() {
        return targetDqnUpdateFreq;
    }

    public void setTargetDqnUpdateFreq(int targetDqnUpdateFreq) {
        this.targetDqnUpdateFreq = targetDqnUpdateFreq;
    }

    public int getUpdateStart() {
        return updateStart;
    }

    public void setUpdateStart(int updateStart) {
        this.updateStart = updateStart;
    }

    public double getRewardFactor() {
        return rewardFactor;
    }

    public void setRewardFactor(double rewardFactor) {
        this.rewardFactor = rewardFactor;
    }

    public double getMinEpsilon() {
        return minEpsilon;
    }

    public void setMinEpsilon(double minEpsilon) {
        this.minEpsilon = minEpsilon;
    }

    public double getGamma() {
        return gamma;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

    public double getErrorClamp() {
        return errorClamp;
    }

    public void setErrorClamp(double errorClamp) {
        this.errorClamp = errorClamp;
    }

    public int getEpsilonNbStep() {
        return epsilonNbStep;
    }

    public void setEpsilonNbStep(int epsilonNbStep) {
        this.epsilonNbStep = epsilonNbStep;
    }

    public boolean isDoubleDQN() {
        return doubleDQN;
    }

    public void setDoubleDQN(boolean doubleDQN) {
        this.doubleDQN = doubleDQN;
    }

    public double getL2() {
        return l2;
    }

    public void setL2(double l2) {
        this.l2 = l2;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getNumHiddenNodes() {
        return numHiddenNodes;
    }

    public void setNumHiddenNodes(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }

    public QLearningConfiguration getQLearningConfiguration(int nWeeks, int windowSize) {

        return QLearningConfiguration.builder()
                .seed(seed)
                .maxEpochStep((int) Math.ceil((double) nWeeks / windowSize))
                .maxStep(maxStep)
                .expRepMaxSize(expRepMaxSize)
                .batchSize(batchSize)
                .targetDqnUpdateFreq(targetDqnUpdateFreq)
                .updateStart(updateStart) // 0
                .minEpsilon(minEpsilon)
                .gamma(gamma)
                .epsilonNbStep(epsilonNbStep)
                .doubleDQN(doubleDQN)
                .errorClamp(errorClamp)     // 1
                .rewardFactor(rewardFactor) // 1
                .build();
    }

    public DQNDenseNetworkConfiguration getNetworkConfiguration() {
        return DQNDenseNetworkConfiguration.builder()
                .l2(l2)
                .updater(new Adam(learningRate))
                .numHiddenNodes(numHiddenNodes)
                .numLayers(numLayers)
                .build();
    }
}
