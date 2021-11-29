import burlap.behavior.functionapproximation.dense.ConcatenatedObjectFeatures;
import burlap.behavior.functionapproximation.dense.NumericVariableFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.state.LLAgent;
import burlap.domain.singleagent.lunarlander.state.LLBlock;
import burlap.domain.singleagent.lunarlander.state.LLState;
import burlap.mdp.singleagent.oo.OOSADomain;
import edu.gatech.cs7641.hw4.DeterministicGridWorldExample;
import edu.gatech.cs7641.hw4.LunarLanderProblem;
import edu.gatech.cs7641.hw4.MountainCarProblem;
import edu.gatech.cs7641.hw4.StochasticGridWorld;
import javafx.util.Pair;


import java.io.*;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Date;


public class Driver {
    private  static final Boolean DETERMINISTIC_MAP = true;
    private static final long DETERMINISTIC_MAP_GENERATION_SEED = 1234567890;
    private static final int SMALL_GW_SIZE = 5;
    private static final int LARGE_GW_SIZE = 25;
    public static void main(String[] args) throws IOException {
        String dateString = new SimpleDateFormat("yyyyMMdd").format(new Date());
        File logFile = new File("mylogfile_" + dateString +".log");
        PrintStream logOut = new PrintStream(new FileOutputStream(logFile, true));

        PrintStream teeStdOut = new TeeStream(System.out, logOut);
        PrintStream teeStdErr = new TeeStream(System.err, logOut);

        System.setOut(teeStdOut);
        System.setErr(teeStdErr);
        //DoSGWStuff();
        //DoLLStuff();

        //DoSGWStuff();
        DoMCStuff();
        //DoTimedMCStuff();

        //DoTimedGWStuff();
    }


    protected static void DoMCStuff(){

        MountainCarProblem mc = new MountainCarProblem(50);
        mc.DoExpirimentComparisons(0.99);
        //mc.DoSinglePI(true,0.99);
        //mc.DoSingleVI(true,0.99);
    }

    protected static void DoSGWStuff(){
        final int x_sz = LARGE_GW_SIZE;
        final int y_sz = LARGE_GW_SIZE;
        final int iter = 1000;
        final double gamma = 0.99;
        final double prob_success = 1.0;
        final double percent_traps = 0.15;
        final double percent_bad_endings=0.03;
        final long map_seed = (DETERMINISTIC_MAP)?DETERMINISTIC_MAP_GENERATION_SEED: Instant.now().toEpochMilli();
        StochasticGridWorld sgw = new StochasticGridWorld(x_sz,y_sz,iter,prob_success,percent_traps,percent_bad_endings,map_seed);
        //sgw.DoSingleValueIteration(true,gamma);
        //sgw.DoSinglePolicyIteration(true,gamma);
        //sgw.DoSingleQLearning(true,gamma);
        //sgw.DoSingleDFS(true);
        //sgw.DoSingleAStar(true);
        //sgw.DoExpirimentComparisons(0.10);
    }
    protected static void DoTimedMCStuff() throws IOException {
        System.out.println("Running timed MountainCar stuff");
        ArrayList<String[]> metrics_table = new ArrayList<>();
        metrics_table.add(new String[]{"domain","solver","time","iter","gamma"});
        double gammas[] = {0.1,0.3,0.7,0.8,0.99,1};
        for (int iter=5; iter<60; iter+=10)
        {
            System.out.printf("setting iterations to %d\n",iter);
            for (double gamma: gammas){
                System.out.printf("Setting gamma to %f\n",gamma);
                MountainCarProblem mc = new MountainCarProblem(iter);
                System.out.println("run value iteration");
                long t = Instant.now().toEpochMilli();
                mc.DoSingleVI(false,gamma);
                long duration = Instant.now().toEpochMilli() - t;
                metrics_table.add(new String[]{
                        "mc",
                        "vi",
                        Long.toString(duration),
                        Integer.toString(iter),
                        Double.toString(gamma),
                        Double.toString(mc.GetBestReturn()),
                        Double.toString(mc.GetAverageReturn())
                });
                System.out.printf("value iteration elapsed %dms\n",duration);
                System.out.println("run policy iteration");
                mc =  new MountainCarProblem(iter);
                t = Instant.now().toEpochMilli();
                mc.DoSinglePI(false,gamma);
                duration = Instant.now().toEpochMilli() - t;
                metrics_table.add(new String[]{
                        "mc",
                        "pi",
                        Long.toString(duration),
                        Integer.toString(iter),
                        Double.toString(gamma),
                        Double.toString(mc.GetBestReturn()),
                        Double.toString(mc.GetAverageReturn())
                });
                System.out.printf("policy iteration elapsed %dms\n",duration);
                System.out.println("run qlearning");
                mc =  new MountainCarProblem(iter);
                t = Instant.now().toEpochMilli();
                mc.DoSingleQL(false,gamma);
                duration = Instant.now().toEpochMilli() - t;
                metrics_table.add(new String[]{
                        "mc",
                        "qlearn",
                        Long.toString(duration),
                        Integer.toString(iter),
                        Double.toString(gamma),
                        Double.toString(mc.GetBestReturn()),
                        Double.toString(mc.GetAverageReturn())
                });
                System.out.printf("qlearning elapsed %dms\n",duration);
            }
        }

        WriteTimedSGWStuffAsCSV(metrics_table);

    }
    protected static void DoTimedGWStuff() throws IOException {
        System.out.println("Running timed Gridworld stuff");
        ArrayList<String[]> metrics_table = new ArrayList<>();
        metrics_table.add(new String[]{"domain","solver","time","dim","iter","gamma","bestscore","avgscore"});
        double gammas[] = {0.1,0.3,0.5,0.8,0.99,1};
        //double gammas[] = {0.99};
        double prob_success = 0.8;
        double percent_traps = 0.3;
        double percent_bad_endings=0.03;
        long map_seed = (DETERMINISTIC_MAP)?DETERMINISTIC_MAP_GENERATION_SEED: Instant.now().toEpochMilli();
        for (int dim : new int[]{SMALL_GW_SIZE,LARGE_GW_SIZE}){
            System.out.printf("Setting dimension to %d\n",dim*dim);
            for (int iter=200; iter<1000; iter+=200)
            {
                System.out.printf("setting iterations to %d\n",iter);
                int x_sz = dim;
                int y_sz = dim;
                for (double gamma: gammas){
                    System.out.printf("Setting gamma to %f\n",gamma);
                    StochasticGridWorld sgw = new StochasticGridWorld(x_sz,y_sz,iter,prob_success,percent_traps,percent_bad_endings,map_seed);

                    System.out.println("run value iteration");
                    long t = Instant.now().toEpochMilli();
                    sgw.DoSingleValueIteration(false,gamma);
                    long duration = Instant.now().toEpochMilli() - t;
                    metrics_table.add(new String[]{
                            "gw",
                            "vi",
                            Long.toString(duration),
                            Integer.toString(dim),
                            Integer.toString(iter),
                            Double.toString(gamma),
                            Double.toString(sgw.GetBestReturn()),
                            Double.toString(sgw.GetAverageReturn())
                    });
                    System.out.printf("value iteration elapsed %dms\n",duration);
                    System.out.println("run policy iteration");
                    sgw = new StochasticGridWorld(x_sz,y_sz,iter,prob_success,percent_traps,percent_bad_endings,map_seed);
                    t = Instant.now().toEpochMilli();
                    sgw.DoSinglePolicyIteration(false,gamma);
                    duration = Instant.now().toEpochMilli() - t;
                    metrics_table.add(new String[]{
                            "gw",
                            "pi",
                            Long.toString(duration),
                            Integer.toString(dim),
                            Integer.toString(iter),
                            Double.toString(gamma),
                            Double.toString(sgw.GetBestReturn()),
                            Double.toString(sgw.GetAverageReturn())
                    });
                    System.out.printf("policy iteration elapsed %dms\n",duration);
                    System.out.println("run qlearning");
                    sgw = new StochasticGridWorld(x_sz,y_sz,iter,prob_success,percent_traps,percent_bad_endings,map_seed);
                    t = Instant.now().toEpochMilli();
                    sgw.DoSingleQLearning(false,gamma);
                    duration = Instant.now().toEpochMilli() - t;
                    metrics_table.add(new String[]{
                            "gw",
                            "qlearn",
                            Long.toString(duration),
                            Integer.toString(dim),
                            Integer.toString(iter),
                            Double.toString(gamma),
                            Double.toString(sgw.GetBestReturn()),
                            Double.toString(sgw.GetAverageReturn())
                    });
                    System.out.printf("qlearning elapsed %dms\n",duration);
                }
            }
        }
        WriteTimedSGWStuffAsCSV(metrics_table);

    }

    protected static void WriteTimedSGWStuffAsCSV(ArrayList<String[]> metrics_table ) throws IOException {
        File fout = new File("metrics.csv");
        FileOutputStream fos = new FileOutputStream(fout);

        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        for (String[] sarr:metrics_table){
            String out_str = String.join(",",sarr);
            bw.write(out_str);
            bw.newLine();
        }

        bw.close();

    }
    static class TeeStream extends PrintStream {
        PrintStream out;
        public TeeStream(PrintStream out1, PrintStream out2) {
            super(out1);
            this.out = out2;
        }
        public void write(byte buf[], int off, int len) {
            try {
                super.write(buf, off, len);
                out.write(buf, off, len);
            } catch (Exception e) {
            }
        }
        public void flush() {
            super.flush();
            out.flush();
        }
    }
}
