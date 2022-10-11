package sample;

import com.google.gson.Gson;
import markovdecisionproblem.ABMEnv;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DQNBrandAgentTest {

    public static void main(String[] args) throws IOException {

        // ModelBuilder configuration ------------------------------------------
        // Paths to files
        Gson gson = new Gson();
        String pathMDPConfig = args[0];
        String pathPolicy = args[1];

        // Get ModelDefinition
        String mdpConfigString = new String(Files.readAllBytes(Paths.get(pathMDPConfig)));
        MDPConfig mdpConfig = gson.fromJson(mdpConfigString,
                MDPConfig.class);

        // Create the ABM environment.
        ABMEnv mdp = mdpConfig.getABMenv(0);

        DQNPolicy<Observation> pol = DQNPolicy.load(pathPolicy);

        // Evaluate the agent
        double rewards = 0;
        int nEpisodes = 15;
        for (int i = 0; i < nEpisodes; i++) {
            double reward = pol.play(mdp);
            rewards += reward;
            System.out.println("Reward: " + reward);
        }

        System.out.println("average: " + rewards / nEpisodes);
        mdp.close();
    }

}
