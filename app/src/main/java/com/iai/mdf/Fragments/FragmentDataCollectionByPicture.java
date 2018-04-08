package com.iai.mdf.Fragments;

import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.Toast;

import com.iai.mdf.DependenceClasses.Configuration;
import com.iai.mdf.DependenceClasses.DeviceProfile;
import com.iai.mdf.Handlers.ImageFileHandler;
import com.iai.mdf.Activities.DataCollectionActivity;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.R;

import java.util.ArrayList;

/**
 * Created by mou on 9/16/17.
 */


public class FragmentDataCollectionByPicture extends Fragment {


    private final String LOG_TAG = "FragmentDataByPicture";
    private View dotHolderLayout;
    private CameraHandler cameraHandler;
    private DrawHandler drawHandler;
    private int[] SCREEN_SIZE;
    private int firstSeveralDots = 0;
    private int dotCounter = 0;
    private boolean isPicSaved = true;
    private boolean isStarted = false;



    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        getActivity().setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
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
                if( isPicSaved ) {
                    isPicSaved = false;
                    dotCounter++;
                    drawHandler.showNextPoint();
                    delayCapture(Configuration.getInstance(getContext()).getCollectionCaptureDelayTime());
                    if( dotCounter < 4 ){
                        cameraHandler.deleteLastPicture();
                    }
                }
                dotGeneratorHandler.postDelayed(this,10);
            }
        };
        dotHolderLayout.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if (!isStarted){
                    isStarted = true;
                    dotGeneratorHandler.postDelayed(dotGeneratorRunnable, 500);
                } else {
                    isStarted = false;
                    dotGeneratorHandler.removeCallbacks(dotGeneratorRunnable);
                    ((FrameLayout)dotHolderLayout).removeAllViews();
                    Toast.makeText(getActivity(), "Take " + String.valueOf(dotCounter-3) + " pictures", Toast.LENGTH_LONG).show();
                    dotCounter = 0;
                    cameraHandler.deleteLastPicture();
                }
                return false;
            }
        });
        Toast.makeText(getActivity(), "Click anywhere to start\nFirst 3 dots don't count", Toast.LENGTH_LONG).show();

    }

    @Override
    public void onResume() {
        super.onResume();
        cameraHandler = new CameraHandler(getActivity(), false);
        cameraHandler.setImageSize(DataCollectionActivity.Image_Size);
        cameraHandler.setSavingCallback(new ImageFileHandler.SavingCallback() {
            @Override
            public void onSaved() {
                isPicSaved = true;
            }
        });
        cameraHandler.openFrontCameraForDataCollection();
        SCREEN_SIZE = fetchScreenSize();
        Log.d(LOG_TAG, "Width: " + SCREEN_SIZE[0] + "    Height: " + SCREEN_SIZE[1]);
        drawHandler = new DrawHandler(getActivity(), SCREEN_SIZE);
        drawHandler.setDotHolderLayout((FrameLayout)dotHolderLayout);
    }


    @Override
    public void onPause(){
        super.onPause();
        if( isPicSaved ){
            cameraHandler.deleteLastPicture();
            firstSeveralDots = 0;
        }
        cameraHandler.closeFrontCameraForDataCollection();
    }


    public boolean isPicSaved(){
        return  isPicSaved;
    }

    private void delayCapture(int delayLength){
        Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                cameraHandler.takePicture(drawHandler.getCurrDot());
            }
        }, delayLength);
    }

    private int[] fetchScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getActivity().getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }


}
