package edu.gatech.cs7641.hw4;

import burlap.behavior.functionapproximation.DifferentiableStateActionValue;
import burlap.behavior.functionapproximation.dense.ConcatenatedObjectFeatures;
import burlap.behavior.functionapproximation.dense.DenseCrossProductFeatures;
import burlap.behavior.functionapproximation.dense.NormalizedVariableFeatures;
import burlap.behavior.functionapproximation.dense.NumericVariableFeatures;
import burlap.behavior.functionapproximation.dense.fourier.FourierBasis;
import burlap.behavior.functionapproximation.dense.rbf.DistanceMetric;
import burlap.behavior.functionapproximation.dense.rbf.RBFFeatures;
import burlap.behavior.functionapproximation.dense.rbf.functions.GaussianRBF;
import burlap.behavior.functionapproximation.dense.rbf.metrics.EuclideanDistance;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.gridset.FlatStateGridder;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.sparsesampling.SparseSampling;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.cartpole.CartPoleVisualizer;
import burlap.domain.singleagent.cartpole.InvertedPendulum;
import burlap.domain.singleagent.cartpole.states.InvertedPendulumState;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.lunarlander.LLVisualizer;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.state.LLAgent;
import burlap.domain.singleagent.lunarlander.state.LLBlock;
import burlap.domain.singleagent.lunarlander.state.LLState;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MCState;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;

import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import jdk.nashorn.internal.ir.Terminal;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class MountainCarProblem {
    int MAX_ITER;
    MountainCar mcGen;
    TerminalFunction tf;
    RewardFunction rf;
    SADomain domain;
    MCState s;
    SimulatedEnvironment env;
    List<Episode> episodes = new ArrayList<>();
    public MountainCarProblem(int iterations){
        this.MAX_ITER = iterations;
        mcGen = new MountainCar();
        tf = new MountainCar.ClassicMCTF(mcGen.physParams.xmax);
        rf =  new GoalBasedRF(tf,100,-0.1);
        mcGen.setRf(rf);
        mcGen.setTf(tf);
        mcGen.physParams.acceleration=mcGen.physParams.acceleration*4;
        mcGen.physParams.gravity=mcGen.physParams.gravity/4;
        domain = mcGen.generateDomain();
        s = new MCState(mcGen.physParams.valleyPos(), 0.);
        env = new SimulatedEnvironment(domain, new MCState(mcGen.physParams.valleyPos(), 0.));
    }

    public void DoSinglePI(Boolean visualize, double gamma){
        /*
        mcGen = new MountainCar();
        tf = new MountainCar.ClassicMCTF(mcGen.physParams.xmax);
        rf =  new GoalBasedRF(tf,100,-0.1);
        mcGen.setRf(rf);
        mcGen.setTf(tf);

        mcGen.physParams.gravity=mcGen.physParams.gravity/8;
        domain = mcGen.generateDomain();
        s = new MCState(mcGen.physParams.valleyPos(), 0.);
        env = new SimulatedEnvironment(domain, new MCState(mcGen.physParams.valleyPos(), 0.));
        StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
        SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
        SARSData dataset = collector.collectNInstances(rStateGen, domain.getModel(),
                5000, 20, null);

        NormalizedVariableFeatures inputFeatures = new NormalizedVariableFeatures()
                .variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
                .variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));

        FourierBasis fb = new FourierBasis(inputFeatures, 4);

        LSPI lspi = new LSPI(domain, 0.99, new DenseCrossProductFeatures(fb, 3), dataset);
        Policy p = lspi.runPolicyIteration(30, 1e-6);
        if (visualize){
            Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
            VisualActionObserver vob = new VisualActionObserver(v);
            vob.initGUI();

            SimulatedEnvironment env = new SimulatedEnvironment(domain, new MCState(mcGen.physParams.valleyPos(), 0.));
            env.addObservers(vob);
        }

        for(int i = 0; i < MAX_ITER; i++){
            Episode e = PolicyUtils.rollout(p, env);
            episodes.add(e);
            env.resetEnvironment();
        }
        System.out.println("Finished");

         */
        MountainCar mcGen = new MountainCar();
        SADomain domain = mcGen.generateDomain();
        StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
        SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
        SARSData dataset = collector.collectNInstances(rStateGen, domain.getModel(), 5000, 20, null);
        NormalizedVariableFeatures inputFeatures = new NormalizedVariableFeatures()
                .variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
                .variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));
        FourierBasis fb = new FourierBasis(inputFeatures, 4);
        LSPI lspi = new LSPI(domain, gamma, new DenseCrossProductFeatures(fb, 3), dataset);
        Policy p = lspi.runPolicyIteration(MAX_ITER, 1e-6);

        SimulatedEnvironment env = new SimulatedEnvironment(domain, new MCState(mcGen.physParams.valleyPos(), 0.));

        for(int i = 0; i < 5; i++){
            Episode e = PolicyUtils.rollout(p, env);
            episodes.add(e);
            env.resetEnvironment();
        }
        System.out.println("Finished");
    }



    public void DoSingleQL(Boolean visualize, double gamma){
        episodes.clear();
        LearningAgent agent = new QLearning(domain, gamma, new SimpleHashableStateFactory(), 0., 1.);
        if (visualize){
            Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
            VisualActionObserver vob = new VisualActionObserver(v);
            vob.initGUI();
            env.addObservers(vob);
        }
        for (int i = 0; i < MAX_ITER; i++) {
            Episode e = agent.runLearningEpisode(env);
            episodes.add(e);
            System.out.println(i + ": timestep=" + e.maxTimeStep()+" return=" + e.discountedReturn(1)+" reward="+e.rewardSequence);
            System.out.println(i + ": return=" + e.discountedReturn(1));
            env.resetEnvironment();
        }

    }
    public void DoSingleVI(Boolean visualize, double gamma){
        episodes.clear();
        LearningAgent agent = new SarsaLam(domain, gamma, new SimpleHashableStateFactory(), 0., 1.,1);
        if (visualize){
            Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
            VisualActionObserver vob = new VisualActionObserver(v);
            vob.initGUI();
            env.addObservers(vob);
        }
        for (int i = 0; i < MAX_ITER; i++) {
            Episode e = agent.runLearningEpisode(env);
            episodes.add(e);
            System.out.println(i + ": timestep=" + e.maxTimeStep()+" return=" + e.discountedReturn(1)+" reward="+e.rewardSequence);
            System.out.println(i + ": return=" + e.discountedReturn(1));
            env.resetEnvironment();
        }
    }


    public void DoExpirimentComparisons(double gamma){

        final int COLS = 2;
        final int W=COLS*400;
        final int L=COLS*200;
        final int H=W*2;
        final int N=5;
        LearningAgentFactory qlFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "QLearning";
            }

            public LearningAgent generateAgent() {
                return new QLearning(domain, gamma, new SimpleHashableStateFactory(), 0.3, 0.1);
            }
        };

        LearningAgentFactory piFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Policy-Iter";
            }

            public LearningAgent generateAgent() {

                MountainCar mcGen = new MountainCar();
                TerminalFunction tf = new MountainCar.ClassicMCTF(mcGen.physParams.xmax);
                RewardFunction rf =  new GoalBasedRF(tf,100,-1.);
                mcGen.setRf(rf);
                mcGen.setTf(tf);

                //mcGen.physParams.gravity=mcGen.physParams.gravity/8;
                SADomain domain = mcGen.generateDomain();
                State s = new MCState(mcGen.physParams.valleyPos(), 0.);
                SimulatedEnvironment env = new SimulatedEnvironment(domain, new MCState(mcGen.physParams.valleyPos(), 0.));
                StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
                SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
                SARSData dataset = collector.collectNInstances(rStateGen, domain.getModel(),
                        5000, 10, null);

                NormalizedVariableFeatures inputFeatures = new NormalizedVariableFeatures()
                        .variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
                        .variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));

                FourierBasis fb = new FourierBasis(inputFeatures, 4);

                LSPI lspi = new LSPI(domain, gamma, new DenseCrossProductFeatures(fb, 3), dataset);
                //lspi.initializeForPlanning(5000);

                return lspi;
            }
        };
        LearningAgentFactory viFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "ValueIter";
            }

            public LearningAgent generateAgent() {
                return new SarsaLam(domain, gamma, new SimpleHashableStateFactory(), 0., 1.,1);
            }
        };

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
                N, MAX_ITER,qlFactory,viFactory,piFactory);

        exp.setUpPlottingConfiguration(W,L,COLS, H,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.CUMULATIVE_REWARD_PER_STEP,
                PerformanceMetric.AVERAGE_EPISODE_REWARD,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE,
                PerformanceMetric.MEDIAN_EPISODE_REWARD
        );
        exp.startExperiment();
        /*
         exp = new LearningAlgorithmExperimenter(env,
                N, 25,piFactory);

        exp.setUpPlottingConfiguration(W,L,COLS, H,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.CUMULATIVE_REWARD_PER_STEP,
                PerformanceMetric.AVERAGE_EPISODE_REWARD,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE,
                PerformanceMetric.MEDIAN_EPISODE_REWARD
        );
        exp.startExperiment();
         exp = new LearningAlgorithmExperimenter(env,
                N, MAX_ITER,viFactory);

        exp.setUpPlottingConfiguration(W,L,COLS, H,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.CUMULATIVE_REWARD_PER_STEP,
                PerformanceMetric.AVERAGE_EPISODE_REWARD,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE,
                PerformanceMetric.MEDIAN_EPISODE_REWARD
        );
        exp.startExperiment();
*/

    }
    public double GetBestReturn(){
        return (episodes.isEmpty()) ?
                0 :
                episodes.stream()
                        .max(Comparator.comparing(e -> e.discountedReturn(1)))
                        .get()
                        .discountedReturn(1);

    }
    public double GetAverageReturn(){
        double r= (episodes.isEmpty()) ?
                0 :
                episodes.stream()
                        .max(Comparator.comparing(e -> e.discountedReturn(1)))
                        .get()
                        .discountedReturn(1);
        return r/episodes.size();

    }
}
