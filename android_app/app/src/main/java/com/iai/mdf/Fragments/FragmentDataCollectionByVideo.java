package com.iai.mdf.Fragments;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.Toast;

import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.R;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by mou on 9/16/17.
 */


public class FragmentDataCollectionByVideo extends Fragment {

    static public final int       DOT_DURATION_IN_SEC = 2;
    private final String LOG_TAG = "FragmentDataByVideo";

    private View dotHolderLayout;
    private CameraHandler cameraHandler;
    private DrawHandler drawHandler;
    private Toast   toast;
    private int[] SCREEN_SIZE;
    private int         firstSeveralDots = 0;
    private int         dotCounter = 0;
    private String      videoName;



    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_data_collection, container, false);
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        dotHolderLayout = getActivity().findViewById(R.id.fragment_data_collection_layout_dotHolder);

        final Handler dotGeneratorHandler = new Handler();
        final Runnable dotGeneratorRunnable = new Runnable() {
            @Override
            public void run() {
                dotCounter++;
                drawHandler.showNextPoint();
                cameraHandler.recordDotPosition(drawHandler.getCurrDot());
                dotGeneratorHandler.postDelayed(this,DOT_DURATION_IN_SEC*1000);
            }
        };
        dotHolderLayout.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if (!cameraHandler.isVideoing()){
                    cameraHandler.startVideo();
                    if( !cameraHandler.isVideoing() ){
                        toast.makeText(getActivity(), "Can't start taking video.", Toast.LENGTH_SHORT).show();
                    } else {
                        toast.cancel();
                        cameraHandler.recordDotPosition(null);
                        dotGeneratorHandler.post(dotGeneratorRunnable);
                    }
                } else {
                    cameraHandler.stopVideo();
                    cameraHandler.initMediaRecorder();
                    dotGeneratorHandler.removeCallbacks(dotGeneratorRunnable);
                    ((FrameLayout)dotHolderLayout).removeAllViews();
                    dotCounter = 0;
                }
                return false;
            }
        });

    }

    @Override
    public void onResume() {
        super.onResume();
        cameraHandler = new CameraHandler(getActivity());
        cameraHandler.openFrontCameraForVideo();    // media recorder is inilized in this function
        SCREEN_SIZE = fetchScreenSize();
        Log.d(LOG_TAG, "Width: " + SCREEN_SIZE[0] + "    Height: " + SCREEN_SIZE[1]);
        drawHandler = new DrawHandler(getActivity(), SCREEN_SIZE);
        drawHandler.setDotHolderLayout((FrameLayout)dotHolderLayout);
        toast = Toast.makeText(getActivity(), "Press anywhere to start and stop", Toast.LENGTH_LONG);
        toast.show();
    }


    @Override
    public void onPause(){
        super.onPause();
        cameraHandler.releaseResource();
    }


    private int[] fetchScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getActivity().getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }


}
