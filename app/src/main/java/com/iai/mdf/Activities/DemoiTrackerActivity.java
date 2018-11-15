package com.iai.mdf.Activities;

import android.content.pm.ActivityInfo;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.TextView;

import com.iai.mdf.FaceDetectionAPI;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.Handlers.ImageProcessHandler;
import com.iai.mdf.Handlers.TensorFlowHandler;
import com.iai.mdf.Handlers.TimerHandler;
import com.iai.mdf.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

//import com.moutaigua.isl_android_gaze.FaceDetectionAPI;

/**
 * Created by Mou on 9/22/2017.
 */

public class DemoiTrackerActivity extends AppCompatActivity {

    private final String LOG_TAG = "DemoiTrackerActivity";
    private CameraHandler cameraHandler;
    private DrawHandler drawHandler;
    private TextureView textureView;
    private Spinner     spinnerView;
    private FrameLayout view_dot_container;
    private FrameLayout frame_gaze_result;
    private FrameLayout frame_bounding_box;
    private TextView    result_board;
    private int[]       SCREEN_SIZE;
    private int[]       TEXTURE_SIZE;
    private FaceDetectionAPI detectionAPI;
    private BaseLoaderCallback openCVLoaderCallback;
    private boolean isRealTimeDetection = false;
    private Handler autoDetectionHandler = new Handler();
    private Runnable autoDetectionRunnable;
    private int     captureInterval = 333;
    private double[]    theFaces = new double[4];
    private TensorFlowHandler tensorFlowHandler;
    private int         mFrameIndex = 0;
    private int         currentClassNum = 4;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_demo);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        getSupportActionBar().hide();

        // init openCV
        initOpenCV();

        SCREEN_SIZE = fetchScreenSize();
        textureView = (TextureView) findViewById(R.id.activity_demo_preview_textureview);
        // ensure texture fill the screen with a certain ratio
        TEXTURE_SIZE = SCREEN_SIZE;
        int expected_height = TEXTURE_SIZE[0]*DataCollectionActivity.Image_Size.getHeight()/DataCollectionActivity.Image_Size.getWidth();
        if( expected_height< TEXTURE_SIZE[1] ){
            TEXTURE_SIZE[1] = expected_height;
        } else {
            TEXTURE_SIZE[0] = TEXTURE_SIZE[1]*DataCollectionActivity.Image_Size.getWidth()/DataCollectionActivity.Image_Size.getHeight();
        }
        textureView.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));


        view_dot_container = (FrameLayout) findViewById(R.id.activity_demo_layout_dotHolder_background);
        frame_gaze_result = (FrameLayout) findViewById(R.id.activity_demo_layout_dotHolder_result);
        drawHandler = new DrawHandler(this, fetchScreenSize());
        drawHandler.setDotHolderLayout(view_dot_container);
//        drawHandler.showAllCandidateDots();

        frame_bounding_box = (FrameLayout) findViewById(R.id.activity_demo_layout_bounding_box);
        frame_bounding_box.bringToFront();
        frame_bounding_box.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                result_board.setText("");
                drawHandler.clear(frame_bounding_box);
                Log.d(LOG_TAG, "pressed");
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
//                isRealTimeDetection = !isRealTimeDetection;
//                if(isRealTimeDetection){
//                    autoDetectionHandler.post(autoDetectionRunnable);
//                } else {
//                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
//                    autoDetectionHandler.removeCallbacks(autoDetectionRunnable);
//                    result_board.setText("Detection is Off\nPress Anywhere to Start");
//                    initFaceArray(theFaces);    // clear saved faces
//                }
            }
        });


        spinnerView = (Spinner) findViewById(R.id.activity_demo_spinner_class_number);
        ArrayList<String> classNumOptions = new ArrayList<>();
        classNumOptions.add("2x2");
        classNumOptions.add("2x3");
        classNumOptions.add("3x3");
        ArrayAdapter<String> spinnerAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, classNumOptions);
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerView.setAdapter(spinnerAdapter);
        spinnerView.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                if(!isRealTimeDetection){
                    String selectedLabel = (String)adapterView.getSelectedItem();
                    switch (selectedLabel){
                        case "2x2":
                            currentClassNum = 4;
                            Log.d(LOG_TAG, "Selected: 2x2");
                        case "2x3":
                            currentClassNum = 6;
                            Log.d(LOG_TAG, "Selected: 2x3");
                        case "3x3":
                            currentClassNum = 9;
                            Log.d(LOG_TAG, "Selected: 3x3");
                    }
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
                Log.d(LOG_TAG, "Selected Nothing ");
            }
        });
        spinnerView.bringToFront();


        result_board = (TextView) findViewById(R.id.activity_demo_txtview_result);
        result_board.setText("Press Anywhere to Start");

        autoDetectionRunnable = new Runnable() {
            @Override
            public void run() {
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
//                drawHandler.clear(frame_bounding_box);
                drawHandler.clear(frame_gaze_result);
                autoDetectionHandler.postDelayed(this, captureInterval);
            }
        };
        initFaceArray(theFaces);
        tensorFlowHandler = new TensorFlowHandler(this);
        tensorFlowHandler.pickModel(TensorFlowHandler.MODEL_ITRACKER_FILE_NAME);



        // load model
        detectionAPI = new FaceDetectionAPI();
        Log.i(LOG_TAG, "Loading face models ...");
        String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        if (!detectionAPI.loadModel(
                "/"+ base + "/Download/face_det_model_vtti.model",
                "/"+ base + "/Download/model_landmark_49_vtti.model"
        )) {
            Log.d(LOG_TAG, "Error reading model files.");
        }
    }


    @Override
    public void onResume() {
        super.onResume();
        cameraHandler = new CameraHandler(this, true);
        cameraHandler.setOnImageAvailableListenerForPrev(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                Image image = reader.acquireNextImage();
                if( cameraHandler.getCameraState()==CameraHandler.CAMERA_STATE_STILL_CAPTURE ) {
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    Log.d(LOG_TAG, "Take a picture");
                    drawHandler.clear(frame_bounding_box);
                    drawHandler.clear(frame_gaze_result);
//                    detectionDemo(image);
                    float[] res = getGazeEstimation(image);
                    if (res!=null) {
                        Log.d(LOG_TAG, "Landscape Location:   ( " + String.valueOf(res[1]) + ", " + String.valueOf(res[0]) + " )");
                    }
                    drawResult(res);
                }
                image.close();
            }
        });
        cameraHandler.startPreview(textureView);
    }

    @Override
    public void onPause(){
        super.onPause();
        cameraHandler.stopPreview();
        autoDetectionHandler.removeCallbacks(autoDetectionRunnable);
        initFaceArray(theFaces);    // clear saved faces
        isRealTimeDetection = false;
    }



    private void initOpenCV(){
        // used when loading openCV4Android
        openCVLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS: {
                        Log.d(LOG_TAG, "OpenCV loaded successfully");
//                    mOpenCvCameraView.enableView();
//                    mOpenCvCameraView.setOnTouchListener(ColorBlobDetectionActivity.this);
                    }
                    break;
                    default: {
                        super.onManagerConnected(status);
                    }
                    break;
                }
            }
        };
        if (!OpenCVLoader.initDebug()) {
            Log.d(LOG_TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, openCVLoaderCallback);
        } else {
            Log.d(LOG_TAG, "OpenCV library found inside package. Using it!");
            openCVLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private void initFaceArray(double[] faceArray){
        for (int i = 0; i < faceArray.length; i++) {
            faceArray[i] = -1;
        }
    }


    private final int InputEyeSize = 224;
    private final int InputFaceSize = 224;
    private final int InputGridSize = 25;
    private float[] getGazeEstimation(Image image){
        TimerHandler.getInstance().tic();
        Mat colorImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC4);
        ImageProcessHandler.getRGBMat(image, colorImg.getNativeObjAddr());
        Log.d(LOG_TAG, "Format Conversion: " + String.valueOf(TimerHandler.getInstance().toc()));
        TimerHandler.getInstance().tic();
        Mat grayImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC1);
        Imgproc.cvtColor(colorImg, grayImg, Imgproc.COLOR_BGR2GRAY);
        Log.d(LOG_TAG, "Color -> Gray: " + String.valueOf(TimerHandler.getInstance().toc()));
        TimerHandler.getInstance().tic();
        // face[0] is col, face[1] is row
        int[] face = detectionAPI.detectFace(grayImg.getNativeObjAddr(), 30, 300, true);
        Log.d(LOG_TAG, "Face Detection: " + String.valueOf(TimerHandler.getInstance().toc()));
        if( face!=null ){
            double[] faceRatio = new double[]{
                    (double)face[0]/grayImg.cols(),
                    (double)face[1]/grayImg.rows(),
                    (double)face[2]/grayImg.cols(),
                    (double)face[3]/grayImg.rows()
            };
            drawHandler.showBoundingBoxInLandscape(faceRatio, TEXTURE_SIZE, frame_bounding_box, true);
        }
        if( face!=null ){
            TimerHandler.getInstance().tic();
            double[] landmarks = detectionAPI.detectLandmarks(grayImg.getNativeObjAddr(), face);
            Log.d(LOG_TAG, "Landmark Detection: " + String.valueOf(TimerHandler.getInstance().toc()));
            if( landmarks!=null ){
                TimerHandler.getInstance().tic();
                Mat eyeCropMat = new Mat(InputEyeSize, InputEyeSize, CvType.CV_8UC4);
                Mat faceCropMat = new Mat(InputFaceSize, InputFaceSize, CvType.CV_8UC4);
                ArrayList<String> tfInputNodes = new ArrayList<>();
                ArrayList<float[]> tfInputs = new ArrayList<>();
                ArrayList<int[]> tfInputSizes = new ArrayList<>();
                float[] tfEyeInputArray = new float[InputEyeSize*InputEyeSize*3];
                float[] tfFaceInputArray = new float[InputFaceSize*InputFaceSize*3];
                TimerHandler.getInstance().tic();
                int[] lEyeRect = ImageProcessHandler.getEyeRegionCropRectForiTracker(landmarks, grayImg.width(), grayImg.height(), true);
                int[] rEyeRect = ImageProcessHandler.getEyeRegionCropRectForiTracker(landmarks, grayImg.width(), grayImg.height(), false);
                int[] faceRect = new int[]{face[0], face[1], face[3], face[2]};
                if ( lEyeRect!=null && rEyeRect!=null ) {
                    ImageProcessHandler.cropSingleRegion(colorImg.getNativeObjAddr(), lEyeRect, new int[]{InputEyeSize,InputEyeSize}, tfEyeInputArray, eyeCropMat.getNativeObjAddr()); // resize[0] is width; resize[1] is height
                    tfInputNodes.add("leftEye");
                    tfInputs.add(tfEyeInputArray);
                    tfInputSizes.add(new int[]{InputEyeSize, InputEyeSize, 3});
                    Imgcodecs.imwrite("/sdcard/Download/ilEye.jpg", eyeCropMat);
                    ImageProcessHandler.cropSingleRegion(colorImg.getNativeObjAddr(), rEyeRect, new int[]{InputEyeSize,InputEyeSize}, tfEyeInputArray, eyeCropMat.getNativeObjAddr()); // resize[0] is width; resize[1] is height
                    tfInputNodes.add("rightEye");
                    tfInputs.add(tfEyeInputArray);
                    Imgcodecs.imwrite("/sdcard/Download/irEye.jpg", eyeCropMat);
                    tfInputSizes.add(new int[]{InputEyeSize, InputEyeSize, 3});
                    ImageProcessHandler.cropSingleRegion(colorImg.getNativeObjAddr(), faceRect, new int[]{InputFaceSize,InputFaceSize}, tfFaceInputArray, faceCropMat.getNativeObjAddr()); // resize[0] is width; resize[1] is height
                    tfInputNodes.add("face");
                    tfInputs.add(tfFaceInputArray);
                    Imgcodecs.imwrite("/sdcard/Download/ifaceEye.jpg", faceCropMat);
                    tfInputSizes.add(new int[]{InputFaceSize, InputFaceSize, 3});
                    // grid
                    int[] gridRect = faceRectToGrid(faceRect, InputGridSize);
                    float[] tfGridInputArray = normGridArray(gridRect);
                    tfInputNodes.add("grid");
                    tfInputs.add(tfGridInputArray);
                    tfInputSizes.add(new int[]{InputGridSize, InputGridSize});
                    Log.d(LOG_TAG, "Model Input Prepare: " + String.valueOf(TimerHandler.getInstance().toc()));
                    TimerHandler.getInstance().tic();
                    float[] cmLoc = tensorFlowHandler.getEstimatedLocation(tfInputNodes, tfInputs, tfInputSizes);
                    Log.d(LOG_TAG, "Model Inference: " + String.valueOf(TimerHandler.getInstance().toc()));
                    Log.d(LOG_TAG, "Inference Result: " + String.valueOf(cmLoc[0]) + ", " + String.valueOf(cmLoc[1]));
                    return tensorFlowHandler.iThackerCM2Loc(cmLoc);
                }
            }
        }
        return null;
    }


    private void drawResult(float[] estimateGaze){
        if( estimateGaze!=null ){
            switch (currentClassNum){
                case 4:
                    drawHandler.draw4ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result, true); break;
                case 6:
                    drawHandler.draw6ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result, true); break;
                case 9:
                    drawHandler.draw9ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result, true); break;
            }
        }
    }




    private int[] faceRectToGrid(int[] rect, int gridSize){
        int[] grid = new int[4];
        float scaleX = rect[3] / (float)640;
        float scaleY = rect[2] / (float)480;
        grid[0] = Math.round(gridSize*scaleX) + 1;
        grid[1] = Math.round(gridSize*scaleY) + 1;
        grid[2] = Math.round(gridSize*scaleX);
        grid[3] = Math.round(gridSize*scaleY);
        return grid;
    }

    private float[] normGridArray(int[] faceGrid){
        float[] arr = new float[InputGridSize*InputGridSize];
        int xMin = faceGrid[0];
        int xMax = Math.min(InputGridSize-1, faceGrid[0]+faceGrid[2]);
        int yMin = faceGrid[1];
        int yMax = Math.min(InputGridSize-1, faceGrid[1]+faceGrid[3]);
        for(int j=yMin; j<=yMax; j++){
            for(int i=xMin; i<=xMax; i++){
                arr[i+InputGridSize*j]=1;
            }
        }
        // normalize
        float sum = 0;
        for(float each : arr){
            sum += each;
        }
        float mean = sum / InputGridSize / InputGridSize;
        // stddev
        float dev = 0;
        for(float each : arr){
            dev += (each - mean) * (each - mean);
        }
        dev =  (float)Math.sqrt((double)dev/(InputGridSize*InputGridSize-1));
        //
        for(int i=0; i < InputGridSize*InputGridSize; i++){
            arr[i] = (arr[i] - mean) / dev;
        }
        return arr;
    }


    private int[] fetchScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }






}
