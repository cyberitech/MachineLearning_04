package edu.gatech.cs7641.hw4;

import burlap.behavior.functionapproximation.dense.*;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.SampleModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;



import java.awt.*;
import java.io.File;
import java.util.*;
import java.util.List;

public class StochasticGridWorld {
    final int MAX_ITER ;
    final int X_SZ ;
    final int Y_SZ ;
    final int TOTAL_CELLS;
    final int GOAL_X ;
    final int GOAL_Y;
    final int START_X = 0;
    final int START_Y = 0;
    final int END_LOC_TYPE = 0;
    final int SANDTRAP_LOC_TYPE = 2;
    final int BAD_END_LOC_TYPE = 1;
    final int TYPE_COUNT = 3; //END_LOC_TYPE, SANDTRAP_LOC_TYPE, BAD_END_LOC_TYPE;

    final double ACTION_COST = -0.1;
    final double SANDTRAP_PENALTY = -0.5;
    final double BAD_END_PENALTY  = -40;
    final double GOOD_END_REWARD = 100;



    Random r;
    GridWorldDomain gg;
    OOSADomain domain;
    GridWorldRewardFunction rf;
    GridWorldTerminalFunction tf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;
    GridAgent agent;

    ArrayList<GridLocation> sand_traps = new ArrayList<>();
    ArrayList<GridLocation> terminal_positions = new ArrayList<>();
    ArrayList<XY> occupied_cells = new ArrayList<>();
    List<Episode> episodes = new ArrayList<>();
    public StochasticGridWorld(int x_sz, int y_sz, int iterations, double prob_success,double percent_traps,double percent_bad_endings,long random_seed){
        X_SZ=x_sz;
        Y_SZ=y_sz;
        MAX_ITER=iterations;
        GOAL_X = X_SZ-1;
        GOAL_Y = Y_SZ-1;
        TOTAL_CELLS = X_SZ*Y_SZ;
        gg = new GridWorldDomain(X_SZ,Y_SZ);
        gg.setProbSucceedTransitionDynamics(prob_success);
        gg.setNumberOfLocationTypes(TYPE_COUNT);
        gg.makeEmptyMap();
        rf = new FixedGridWorldRewardFunction(X_SZ,Y_SZ,ACTION_COST);
        tf = new GridWorldTerminalFunction();
        hashingFactory = new SimpleHashableStateFactory();
        r = new Random(random_seed);
        set_start();
        set_goal();
        set_sandtraps(percent_traps);
        set_bad_endings(percent_bad_endings);
        gg.setTf(tf);
        gg.setRf(rf);
        goalCondition = new TFGoalCondition(tf);
        domain = gg.generateDomain();
        initialState = make_initial_state();
        env = new SimulatedEnvironment(domain, initialState);




    }
    protected void set_start(){
        agent = new GridAgent(START_X, START_Y);
        occupied_cells.add(new XY(START_X, START_Y));
    }
    protected GridWorldState make_initial_state(){
        ArrayList<GridLocation> all_locations = new ArrayList<>();
        all_locations.addAll(sand_traps);
        all_locations.addAll(terminal_positions);
        return new GridWorldState(agent,all_locations);
    }
    protected void set_goal(){
        String loc_name = "goal";
        GridLocation gloc = new GridLocation(GOAL_X,GOAL_Y,END_LOC_TYPE,loc_name);
        terminal_positions.add(gloc);
        occupied_cells.add(new XY(GOAL_X,GOAL_Y));
        tf.markAsTerminalPosition(GOAL_X,GOAL_Y);
        rf.setReward(GOAL_X,GOAL_Y,GOOD_END_REWARD);
    }
    protected void set_sandtraps(double percent_traps){
        int num_traps = (int)(TOTAL_CELLS*percent_traps);

        for (int i=0; i<num_traps; i++){
            Integer x_loc = r.nextInt(X_SZ);
            Integer y_loc = r.nextInt(Y_SZ);
            if (is_occupied_cell(x_loc,y_loc))  // if this is a coordinate which is either a beginning or end point, then we need a do-over
                i--;
            else{
                rf.setReward(x_loc,y_loc,SANDTRAP_PENALTY);
                String loc_name = String.format("sandtrap%03d",i);
                GridLocation gloc = new GridLocation(x_loc,y_loc,SANDTRAP_LOC_TYPE,loc_name);
                sand_traps.add(gloc);
                occupied_cells.add(new XY(x_loc,y_loc));
            }
        }
    }
    protected void set_bad_endings(double percent_bad_endings){
        int num_bad_ends = (int)(TOTAL_CELLS*percent_bad_endings);
        for (int i=0; i<num_bad_ends; i++){
            Integer x_loc = r.nextInt(X_SZ);
            Integer y_loc = r.nextInt(Y_SZ);
            if (is_occupied_cell(x_loc,y_loc)){ // if this is a coordinate which is either a beginning or end point, then we need a do-over
                i--;
            }
            else {
                tf.markAsTerminalPosition(x_loc,y_loc);
                rf.setReward(x_loc,y_loc,BAD_END_PENALTY);
                String loc_name = String.format("bad_end%03d",i);
                GridLocation gloc = new GridLocation(x_loc,y_loc,BAD_END_LOC_TYPE,loc_name);
                terminal_positions.add(gloc);
                occupied_cells.add(new XY(x_loc,y_loc));
            }
        }
    }
    public void DoSingleQLearning(Boolean visualize,double gamma) {
        episodes.clear();
        LearningAgent agent = new QLearning(domain, gamma, hashingFactory, 0., 1.);
        for (int i = 0; i < MAX_ITER; i++) {
            Episode e = agent.runLearningEpisode(env);
            episodes.add(e);
            System.out.println(i + ": timestep=" + e.maxTimeStep()+" return=" + e.discountedReturn(1)+" reward="+e.rewardSequence);
            System.out.println(i + ": return=" + e.discountedReturn(1));
            env.resetEnvironment();
        }
        if (visualize){
            Visualizer v = GridWorldVisualizer.getVisualizer(gg.getMap());
            new EpisodeSequenceVisualizer(v, domain, episodes);
        }
        env.resetEnvironment();
    }
    public void DoSingleDFS(Boolean visualize){
        DeterministicPlanner planner = new DFS(domain, goalCondition, hashingFactory);
        Policy p = planner.planFromState(initialState);
        Episode e = PolicyUtils.rollout(p, initialState, domain.getModel());
        episodes.add(e);

        System.out.println("timestep=" + e.maxTimeStep()+"\nreturn=" + e.discountedReturn(1)+"\nreward="+e.rewardSequence);
        System.out.println("return=" + e.discountedReturn(1));
        if (visualize){
            Visualizer v = GridWorldVisualizer.getVisualizer(gg.getMap());
            new EpisodeSequenceVisualizer(v, domain, episodes);
        }
        env.resetEnvironment();
    }
    public void DoSinglePolicyIteration(Boolean visualize,double gamma){
        episodes.clear();
        int n_samples = X_SZ*Y_SZ*10;
        Planner planner = new PolicyIteration(domain, gamma, hashingFactory,10,10,n_samples,MAX_ITER);
        Policy p = planner.planFromState(initialState);
        Episode e = PolicyUtils.rollout(p, initialState, domain.getModel());
        episodes.add(e);
        System.out.println("\ntimestep=" + e.maxTimeStep()+"\nreturn=" + e.discountedReturn(1)+"\nreward="+e.rewardSequence);
        System.out.println("\nreturn=" + e.discountedReturn(1));
        if (visualize){
            manualValueFunctionVis((ValueFunction)planner, p);
            Visualizer v = GridWorldVisualizer.getVisualizer(gg.getMap());
            new EpisodeSequenceVisualizer(v, domain, episodes);
        }
    }
    public void DoSingleValueIteration(Boolean visualize,double gamma) {
        episodes.clear();
        Planner planner = new ValueIteration(domain, gamma, hashingFactory, 10, MAX_ITER);
        Policy p = planner.planFromState(initialState);
        Episode e = PolicyUtils.rollout(p, initialState, domain.getModel());
        episodes.add(e);
        if (visualize){
            manualValueFunctionVis((ValueFunction)planner, p);
            Visualizer v = GridWorldVisualizer.getVisualizer(gg.getMap());
            new EpisodeSequenceVisualizer(v, domain, episodes);
        }
        env.resetEnvironment();
    }
    public void DoSingleAStar(Boolean visualize){
        episodes.clear();
        Heuristic mdistHeuristic = new Heuristic() {
            public double h(State s) {
                GridAgent a = ((GridWorldState)s).agent;
                double mdist = Math.abs(a.x-GOAL_X) + Math.abs(a.y-GOAL_Y);
                return -mdist;
            }
        };

        DeterministicPlanner planner = new AStar(domain, goalCondition, hashingFactory,
                mdistHeuristic);
        Policy p = planner.planFromState(initialState);
        Episode e =  PolicyUtils.rollout(p, initialState, domain.getModel());
        episodes.add(e);
        System.out.println("\ntimestep=" + e.maxTimeStep()+"\nreturn=" + e.discountedReturn(1)+"\nreward="+e.rewardSequence);
        System.out.println("\nreturn=" + e.discountedReturn(1));
        if (visualize){
            Visualizer v = GridWorldVisualizer.getVisualizer(gg.getMap());
            new EpisodeSequenceVisualizer(v, domain, episodes);
        }
        env.resetEnvironment();
    }

    public void manualValueFunctionVis(ValueFunction valueFunction, Policy p){

        List<State> allStates = StateReachability.getReachableStates(
                initialState, domain, hashingFactory);

        //define color function
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        //define a 2D painter of state values, specifying
        //which variables correspond to the x and y coordinates of the canvas
        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setXYKeys("agent:x", "agent:y",
                new VariableDomain(0, X_SZ+1), new VariableDomain(0, Y_SZ+1),
                1, 1);

        //create our ValueFunctionVisualizer that paints for all states
        //using the ValueFunction source and the state value painter we defined
        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, valueFunction);

        //define a policy painter that uses arrow glyphs for each of the grid world actions
        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYKeys("agent:x", "agent:y",
                new VariableDomain(0, X_SZ+1), new VariableDomain(0, Y_SZ+1),
                1, 1);

        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
        spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


        //add our policy renderer to it
        gui.setSpp(spp);
        gui.setPolicy(p);

        //set the background color for places where states are not rendered to grey
        gui.setBgColor(Color.GRAY);

        //start it
        gui.initGUI();


    }

    protected Boolean is_occupied_cell(int x, int y){
        for (XY xy: occupied_cells)
            if (xy.equals(x,y))
                return true;
        return false;
    }
    public void DoExpirimentComparisons(double gamma){

        final int COLS = 2;
        final int W=COLS*400;
        final int L=COLS*200;
        final int H=W*2;
        final int N=5;
        LearningAgentFactory viFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Value-Iter";
            }

            public LearningAgent generateAgent() {
                //
                Planner planner = new ValueIteration(domain,gamma,hashingFactory,10,MAX_ITER);
                Policy p = planner.planFromState(initialState);
                return new SarsaLam(domain, gamma,hashingFactory,0.0,0.5,p,MAX_ITER,0.3);
            }
        };

        LearningAgentFactory piFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Policy-Iter";
            }

            public LearningAgent generateAgent() {
                //int n_samples = X_SZ*Y_SZ*10;
                int n_samples=1000;
                Planner planner = new PolicyIteration(domain, gamma, hashingFactory,1,1,n_samples,MAX_ITER);
                Policy p = planner.planFromState(initialState);
                return new SarsaLam(domain, gamma,hashingFactory,0.0,0.5,p,MAX_ITER,0.3);
            }
        };

        LearningAgentFactory qlFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-Learning";
            }

            public LearningAgent generateAgent() {
                return new QLearning(domain, gamma, hashingFactory, 0.3, 0.1);
            }
        };
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
            N, MAX_ITER, piFactory,viFactory,qlFactory);

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
    //only when proba_success = 1, otherwise this function is kinda useless
    protected Boolean is_done_learning(Episode e1, Episode e2){
        return e1.actionString().equals(e2.actionString());
    }
}
class XY{
    final int x;
    final int y;
    public XY(int x, int y){this.x=x; this.y=y;}
    public Boolean equals(int x, int y){return (this.x == x && this.y==y);}
    public Boolean equals(XY xy){ return (xy.x == x  && xy.y == y); }
}