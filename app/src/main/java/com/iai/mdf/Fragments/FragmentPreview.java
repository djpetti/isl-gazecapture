package com.iai.mdf.Fragments;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.support.v4.app.Fragment;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.app.FragmentTransaction;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;

import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Activities.DataCollectionActivity;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.R;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Created by mou on 9/16/17.
 */


public class FragmentPreview extends Fragment{


    private final String LOG_TAG = "FragmentPreview";
    private CameraHandler cameraHandler;
    private TextureView textureView;
    private OnActionListener onActionListener;
    private DrawHandler drawHandler;
    private FrameLayout frameBackground;
    private Button      btnPicture;
    private Button      btnVideo;
    private View.OnClickListener buttonClickListener;



    public interface OnActionListener{
        void onClick();
    }


    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_preview, container, false);
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        textureView = getActivity().findViewById(R.id.fragment_preview_textureview);
//        textureView.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                onActionListener.onClick();
//            }
//        });
        // ensure texture fill the screen with a certain ratio
        int[] textureSize = fetchScreenSize();
//        int[] textureSize = computeTextureSize(screenSize);
        int expected_height = textureSize[0]*DataCollectionActivity.Image_Size.getHeight()/DataCollectionActivity.Image_Size.getWidth();
        if( expected_height< textureSize[1] ){
            textureSize[1] = expected_height;
        } else {
            textureSize[0] = textureSize[1]*DataCollectionActivity.Image_Size.getWidth()/DataCollectionActivity.Image_Size.getHeight();
        }
        RelativeLayout.LayoutParams params = new RelativeLayout.LayoutParams(textureSize[0], textureSize[1]); // You might want to tweak these to WRAP_CONTENT
//        RelativeLayout.LayoutParams params = new RelativeLayout.LayoutParams(1200, 1600); // You might want to tweak these to WRAP_CONTENT
        params.addRule(RelativeLayout.ALIGN_PARENT_TOP);
        params.addRule(RelativeLayout.CENTER_HORIZONTAL);
        textureView.setLayoutParams(params);

//        // Background Dots Array
//        drawHandler = new DrawHandler(getActivity(), fetchScreenSize());
//        frameBackground = getActivity().findViewById(R.id.fragment_preview_layout_dotHolder_background);
//        drawHandler.showAllCandidateDots(frameBackground);
        btnPicture = getActivity().findViewById(R.id.fragment_preview_btn_picture);
        btnPicture.setOnClickListener(buttonClickListener);
        btnVideo = getActivity().findViewById(R.id.fragment_preview_btn_video);
        btnVideo.setOnClickListener(buttonClickListener);
    }




    @Override
    public void onResume() {
        super.onResume();
        cameraHandler = CameraHandler.getInstance(getActivity(), false);
        cameraHandler.startPreview(textureView);
    }

    @Override
    public void onPause(){
        super.onPause();
        cameraHandler.stopPreview();
    }


    public void setOnActionListener(OnActionListener listener){
        this.onActionListener = listener;
    }

    public void setButtonClickListener(View.OnClickListener clickListener){
        buttonClickListener = clickListener;
    }

    private int[] fetchScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getActivity().getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }


    private int[] computeTextureSize(int[] screenSize){
        int[] textureSize = new int[]{480, 640};
        CameraManager cameraManager = (CameraManager) getActivity().getSystemService(Context.CAMERA_SERVICE);
        String frontCameraId = "unknown";
        try {
            for (final String cameraId : cameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
                if (characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) {
                    frontCameraId = cameraId;
                }
            }
//            CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics("0");
            CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(frontCameraId);
            StreamConfigurationMap map = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size[] allSizes = map.getOutputSizes(SurfaceTexture.class);
            List<Size> collectorSizes = new ArrayList<>();
            // looking for the exact size or the one with the exact ratio;
            double preferredRatio = (double) 640 / 480;
            for(Size option : allSizes) {
                double curRatio = (double)option.getWidth()/option.getHeight();
                if(Math.abs(preferredRatio-curRatio) < 0.000001) {
                    collectorSizes.add(option);
                }
            }
            for(int i = 0; i<collectorSizes.size(); i++){
                if( collectorSizes.get(i).getWidth()<= screenSize[1] &&
                        collectorSizes.get(i).getHeight()<= screenSize[0] ){
                    textureSize[0] = collectorSizes.get(i).getHeight();
                    textureSize[1] = collectorSizes.get(i).getWidth();
                    break;
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        return textureSize;
    }




}
