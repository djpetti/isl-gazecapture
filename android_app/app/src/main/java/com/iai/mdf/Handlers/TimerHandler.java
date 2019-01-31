package com.iai.mdf.Handlers;


/**
 * Created by Mou on 10/31/2017.
 */

public class TimerHandler {


    private final String LOG_TAG = "TimerHandler";
    private static TimerHandler myInstance;
    private long curr_time;


    private TimerHandler(){
        curr_time = 0;
    }

    public static synchronized TimerHandler getInstance(){
        if( myInstance==null ){
            myInstance = new TimerHandler();
        }
        return myInstance;
    }


    public void reset(){
        curr_time = 0;
    }

    public void tic(){
        curr_time = System.currentTimeMillis();
    }

    public long toc(){
        return System.currentTimeMillis() - curr_time;
    }


}
