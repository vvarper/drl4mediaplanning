package sample;

import beans.DQNConfig;
import com.google.gson.Gson;
import markovdecisionproblem.ABMEnv;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DQNBrandAgentTraining {

    public static void main(String[] args) throws IOException {

        // Read configuration files  -------------------------------------------
        // Paths to files
        Gson gson = new Gson();
        String pathMDPConfig = args[0];
        String pathDQNConfig = args[1];
        String pathResultPolicy = args[2];

        // Get configurations
        String mdpConfigString = new String(Files.readAllBytes(Paths.get(pathMDPConfig)));
        String dqnConfigString = new String(Files.readAllBytes(Paths.get(pathDQNConfig)));
        MDPConfig mdpConfig = gson.fromJson(mdpConfigString,
                MDPConfig.class);
        DQNConfig dqnConfig = gson.fromJson(dqnConfigString,
                DQNConfig.class);

        // Q learning configuration --------------------------------------------
        QLearningConfiguration learningConfiguration =
                dqnConfig.getQLearningConfiguration(
                        mdpConfig.getSimulationConfig().getnWeeks(),
                        mdpConfig.getWindowSize()
                );

        // The neural network used by the agent. Note that there is no need to
        // specify the number of inputs/outputs.
        // These will be read from the ABM environment at the start of
        // training.
        DQNDenseNetworkConfiguration network =
                dqnConfig.getNetworkConfiguration();

        // Create the ABM environment
        ABMEnv mdp = mdpConfig.getABMenv(0);

        // Create the solver.
        QLearningDiscreteDense<Observation> dql =
                new QLearningDiscreteDense<>(mdp, network, learningConfiguration);

        System.out.println("Start of training ");
        long startTime = System.nanoTime();
        dql.train();
        long endTime = System.nanoTime();
        System.out.println("End of training");
        long timeElapsed = endTime - startTime;
        System.out.println("Execution time in seconds  : " + timeElapsed / 1e9);

        mdp.close();

        DQNPolicy<Observation> pol = dql.getPolicy();

        pol.save(pathResultPolicy);
    }
}
