package edu.gatech.cs7641.hw4;

import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

/*
*  The GridWorldReward function that comes with Burlap is flawed.
*  The reward(s,a,sprime) method erroneously returns the reward of the cell the agent just LEFT not where it just transitioned TO
*  So i fixed it.
 */
public class FixedGridWorldRewardFunction extends GridWorldRewardFunction {
    public FixedGridWorldRewardFunction(int width, int height, double initializingReward){
        super(width,height,initializingReward);
    }
    public FixedGridWorldRewardFunction(int width, int height){
        super(width, height, 0.);
    }

    @Override
    public double reward(State s, Action a, State sprime) {

        //int x = ((GridWorldState)s).agent.x;
        //int y = ((GridWorldState)s).agent.y;
        int x = ((GridWorldState)sprime).agent.x;    //  s is obviously incorrect, i changed it to sprime.
        int y = ((GridWorldState)sprime).agent.y;

        if(x >= this.width || x < 0 || y >= this.height || y < 0){
            throw new RuntimeException("GridWorld reward matrix is only defined for a " + this.width + "x" +
                    this.height +" world, but the agent transitioned to position (" + x + "," + y + "), which is outside the bounds.");
        }

        double r = this.rewardMatrix[x][y];
        return r;
    }


}
