package com.iai.mdf.Activities;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.TextView;


import com.iai.mdf.FaceDetectionAPI;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

/**
 * Created by Mou on 11/12/2017.
 */

public class OpenCVActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{

    private final String LOG_TAG = "OpenCVActivity";
    private Mat                    mRgba;
    private Mat                    mGray;
    private int                    mFrameIndex = 0;
    private CameraBridgeViewBase mOpenCvCameraView;
    private FaceDetectionAPI mAPI = new FaceDetectionAPI();
//    private TextView            res_board;
//    private FrameLayout         graphHolder;
//    private DrawHandler         drawHandler;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_opencv);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.activity_opencv_view_HelloOpenCvView);
        mOpenCvCameraView.setCameraIndex(1);    //front camera
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableView();

//        res_board = (TextView) findViewById(R.id.activity_opencv_txt_resboard);
//        graphHolder = (FrameLayout) findViewById(R.id.activity_opencv_layout_graph_panel);
//        drawHandler = new DrawHandler(OpenCVActivity.this, new int[]{0,0});

        // Load model

        Log.i(LOG_TAG, "Loading face models ...");
        String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        if (!mAPI.loadModel(
                "/"+ base + "/Download/face_det_model_vtti.model",
                "/"+ base + "/Download/model_landmark_49_vtti.model"
        )) {
            Log.d(LOG_TAG, "Error reading model files.");
        }


        File file = new File("/sdcard/Download/opencvTest/faces.txt");
        if( file.exists() ) {
            file.delete();
        }

    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(LOG_TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        mFrameIndex = 0;
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // Update frame index

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
//        Core.flip(mGray.t(), mGray, 0);
        long matAddrGray = mGray.getNativeObjAddr();
        long matAddrRgba = mRgba.getNativeObjAddr();


        // Face detection
        int[] face = null;
        if (true) {
            face = mAPI.detectFace(matAddrGray, 30, 180, false);
//            detectionResult.setFace(face);
        }

        // Landmark extraction
        double[] landmarks = null;
        if (true && face != null) {
            landmarks = mAPI.detectLandmarks(matAddrGray, face);
//            detectionResult.setLandmarks(landmarks);
        }

        // Draw results
        if (true && face != null) {
            mAPI.drawBoundingBox(matAddrRgba, face);
        }
        if (true && landmarks != null) {
            mAPI.drawLandmarks(matAddrRgba, landmarks);
        }

//        Log.d(LOG_TAG, "Image Availble");
//        if( face!=null ) {
//            mFrameIndex ++;
//            File album = new File("/sdcard/Download/opencvTest/");
//            if( album.isDirectory() || album.mkdir() ){
//                Imgcodecs.imwrite("/sdcard/Download/opencvTest/frame"+String.valueOf(mFrameIndex)+".jpg", mGray);
//                String faceString = String.valueOf(mFrameIndex)+ " "
//                        + String.valueOf(face[0])+ " "
//                        + String.valueOf(face[1])+ " "
//                        + String.valueOf(face[2])+ " "
//                        + String.valueOf(face[3]) + "\n";
//                try {
//                    FileWriter fileWriter = new FileWriter("/sdcard/Download/opencvTest/faces.txt", true);
//                    fileWriter.append(faceString);
//                    fileWriter.close();
//                }
//                catch (IOException e) {
//                    Log.e("Exception", "File write failed: " + e.toString());
//                }
//            } else {
//                Log.d(LOG_TAG, "Directory doesn't exist");
//            }
//
//        }

        // Landmark extraction

//        double[] landmarks = null;
//
//        if (face != null) {
//            String faceDetectionRes = "{" +
//                    String.valueOf(face[0]) + ", " +
//                    String.valueOf(face[1]) + ", " +
//                    String.valueOf(face[2]) + ", " +
//                    String.valueOf(face[3]) + "]";
//            Log.d(LOG_TAG, faceDetectionRes);
//            res_board.setText(faceDetectionRes);
//            drawHandler.showRect(face[0], face[1], face[2], face[3], graphHolder);
//            landmarks = mAPI.detectLandmarks(matAddrGray, face);
//
//        }







        return mRgba;
    }
}
