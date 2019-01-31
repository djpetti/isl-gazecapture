package com.iai.mdf.Activities;

import android.content.pm.ActivityInfo;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.iai.mdf.DependenceClasses.DeviceConfiguration;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.Handlers.SocketHandler;
import com.iai.mdf.Handlers.TensorFlowHandler;
import com.iai.mdf.R;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;

import java.util.ArrayList;

//import com.moutaigua.isl_android_gaze.FaceDetectionAPI;

/**
 * Created by Mou on 9/22/2017.
 */

public class DemoServerActivity2 extends AppCompatActivity {

    public static final String BUNDLE_KEY_IP = "ip";
    public static final String BUNDLE_KEY_PORT = "port";
    private final String LOG_TAG = "DemoServerActivity2";


    private CameraHandler cameraHandler;
    private DrawHandler drawHandler;
    private TextureView textureView;
    private ToggleButton toggleButton;
    private FrameLayout textureviewHolder;
    private FrameLayout frame_background_grid;
    private FrameLayout view_dot_container;
    private FrameLayout frame_gaze_result;
    private FrameLayout frame_bounding_box;
    private TextView    result_board;
    private int[]       SCREEN_SIZE;
    private int[]       TEXTURE_SIZE;
    private BaseLoaderCallback openCVLoaderCallback;
    private boolean     isRealTimeDetection = false;
    private Handler     autoDetectionHandler = new Handler();
    private Runnable    takePicRunnable;
    private Runnable    autoDotGenerationRunnable;
    private TensorFlowHandler tensorFlowHandler;
    private int         mFrameIndex = 0;
    private int         currentClassNum = 4;
    private SocketHandler socketHandler;
    private int         prevReceivedGazeIndex = 0;
    private String      socketIp = null;
    private int         socketPort;
    private ArrayList<Point> estimationList = new ArrayList<>();
    private DeviceConfiguration confHandler = DeviceConfiguration.getInstance(this);



    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_demo_2);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getSupportActionBar().hide();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        Bundle extras = getIntent().getExtras();
        socketIp = extras.getString(BUNDLE_KEY_IP);
        socketPort = extras.getInt(BUNDLE_KEY_PORT);

        // init openCV
        initOpenCV();

        SCREEN_SIZE = fetchScreenSize();
        textureView = findViewById(R.id.activity_demo_preview_textureview);
        textureView.setRotation((float) 270.0);
        // ensure texture fill the screen with a certain ratio
        TEXTURE_SIZE = new int[] { SCREEN_SIZE[0], SCREEN_SIZE[1] };
        int imageWidth = Math.max(DataCollectionActivity.Image_Size.getHeight(), DataCollectionActivity.Image_Size.getWidth());
        int imageHeight = Math.min(DataCollectionActivity.Image_Size.getHeight(), DataCollectionActivity.Image_Size.getWidth());
        int expected_width = TEXTURE_SIZE[1] * imageWidth / imageHeight;
        if( expected_width < TEXTURE_SIZE[0] ){
            TEXTURE_SIZE[0] = expected_width;
        } else {
            TEXTURE_SIZE[1] = TEXTURE_SIZE[0] * imageHeight / imageWidth;
        }
        textureviewHolder = findViewById(R.id.activity_demo_layout_textureview_holder);
        textureviewHolder.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                TEXTURE_SIZE[1], TEXTURE_SIZE[0]
        );
        params.gravity = Gravity.CENTER;
        textureView.setLayoutParams(params);


        view_dot_container = findViewById(R.id.activity_demo_layout_dotHolder_background);
        frame_background_grid = findViewById(R.id.activity_demo_layout_background_grid);
//        frame_background_grid.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));
//        frame_background_grid.bringToFront();
        frame_gaze_result = findViewById(R.id.activity_demo_layout_dotHolder_result);
        frame_gaze_result.setLayoutParams(new RelativeLayout.LayoutParams(SCREEN_SIZE[0], SCREEN_SIZE[1]));
        frame_gaze_result.bringToFront();
        drawHandler = new DrawHandler(this, fetchScreenSize());
        drawHandler.setDotHolderLayout(view_dot_container);
//        drawHandler.showAllCandidateDots();

        frame_bounding_box = findViewById(R.id.activity_demo_layout_bounding_box);
//        frame_bounding_box.bringToFront();
        frame_bounding_box.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawHandler.clear(frame_bounding_box);
                drawHandler.clear(frame_gaze_result);
                drawHandler.clear(view_dot_container);
                Log.d(LOG_TAG, "pressed");  //cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
                if( !socketHandler.isConnected() && !isRealTimeDetection ){
                    Toast.makeText(DemoServerActivity2.this, "Restart to connect to the server", Toast.LENGTH_SHORT).show();
                    return;
                }
                isRealTimeDetection = !isRealTimeDetection;
                if(isRealTimeDetection){
                    frame_background_grid.setBackgroundColor(0xFFFFFFFF);   // cover texture with white
                    autoDetectionHandler.post(takePicRunnable);
                    autoDetectionHandler.post(autoDotGenerationRunnable);
                    result_board.setText("");
                } else {
                    frame_background_grid.setBackgroundColor(0x00FFFFFF);   // uncover texture with translucent
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    autoDetectionHandler.removeCallbacks(takePicRunnable);
                    autoDetectionHandler.removeCallbacks(autoDotGenerationRunnable);
                    result_board.setText("Press Anywhere to Start");
                }
            }
        });

        toggleButton = (ToggleButton) findViewById(R.id.activity_demo_toggle_show_dot);
        toggleButton.setChecked(false);
        toggleButton.setTextOn("stabilize");
        toggleButton.setTextOff("stabilize");

        result_board = (TextView) findViewById(R.id.activity_demo_txtview_result);
        result_board.setText("Press Anywhere to Start");

        takePicRunnable = new Runnable() {
            @Override
            public void run() {
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
                autoDetectionHandler.postDelayed(this, confHandler.getDemoCaptureDelayTime());
            }
        };
        autoDotGenerationRunnable = new Runnable() {
            @Override
            public void run() {
                drawHandler.clear(view_dot_container);
                drawHandler.drawRandomBlockInCandidates(80,80, view_dot_container, true);
                autoDetectionHandler.postDelayed(this, 2000);
                if( toggleButton.isChecked() ) {
                    estimationList.clear();
                }
            }
        };

        tensorFlowHandler = new TensorFlowHandler(this);
        tensorFlowHandler.pickModel(TensorFlowHandler.MODEL_ISL_FILE_NAME);
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
//                    drawHandler.clear(frame_bounding_box);
//                    drawHandler.clear(frame_gaze_result);
                    if( Build.MODEL.equalsIgnoreCase("BLU Studio Touch")) {
                        socketHandler.uploadImageOnBLU(image);
                    } else {
                        socketHandler.uploadImage(image, confHandler);
                    }
                }
                image.close();
            }
        });
        cameraHandler.startPreview(textureView);
        // init socket communication
        initSocketConnection();
    }

    @Override
    public void onPause(){
        super.onPause();
        cameraHandler.stopPreview();
        autoDetectionHandler.removeCallbacks(takePicRunnable);
        autoDetectionHandler.removeCallbacks(autoDotGenerationRunnable);
        isRealTimeDetection = false;
        // stop socket communication
        socketHandler.socketDestroy();
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

    private void initSocketConnection(){
        socketHandler = new SocketHandler(socketIp, socketPort);
        socketHandler.setUiThreadHandler(new SocketHandler.StringCallback() {
            @Override
            public void onResponse(String str) {
                try {
                    if (str.equalsIgnoreCase(SocketHandler.SUCCESS_CONNECT_MSG)){
                        return;
                    }
                    JSONObject object = new JSONObject(str);
                    if (object != null) {
                        if( object.getBoolean(SocketHandler.JSON_KEY_VALID) && isRealTimeDetection ) {
                            drawGaze(object);
                            Log.d(LOG_TAG, object.toString());
                        } else {
                            Log.d(LOG_TAG, "inValid");
                        }
                    }
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }

            @Override
            public void onError(String str) {
                Log.e(LOG_TAG, str);
                if( str.equalsIgnoreCase(SocketHandler.ERROR_DISCONNECTED) ){
                    Toast.makeText(DemoServerActivity2.this, "Disconnect From Server\nRestart Please", Toast.LENGTH_SHORT).show();
                    frame_bounding_box.performClick();
                } else if (str.equalsIgnoreCase(SocketHandler.ERROR_TIMEOUT)) {
                     Log.d(LOG_TAG, "Timeout");
                } else if (str.equalsIgnoreCase(SocketHandler.ERROR_SETTING)) {
                    Toast.makeText(DemoServerActivity2.this, "Please set the address and the port correctly", Toast.LENGTH_SHORT).show();
                    result_board.setText("Wrong address or port");
                }
            }
        });
        socketHandler.socketCreate();
    }







    private int[] fetchScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }

    private void drawGaze(JSONObject object){
        try {
//            int receivedIdx = object.getInt(SocketHandler.JSON_KEY_SEQ_NUMBER);
//            if( receivedIdx > prevReceivedGazeIndex ){
//            prevReceivedGazeIndex = receivedIdx;
            double landscapeHori = object.getDouble(SocketHandler.JSON_KEY_PREDICT_X);
            double landscapeVert = object.getDouble(SocketHandler.JSON_KEY_PREDICT_Y);
            float[] loc = new float[2];
            loc[0] = (float) (landscapeHori + confHandler.getCameraOffsetPHeight())/confHandler.getScreenSizePWidth();
            loc[1] = 1 - (float) (landscapeVert + confHandler.getCameraOffsetPWidth())/confHandler.getScreenSizePHeight();
            // linear calibration
            float[] mat = confHandler.getCalibrationMatrix();
            loc[0] = loc[0] * mat[0]  + loc[1] * mat[2] + mat[4];
            loc[1] = loc[0] * mat[1]  + loc[1] * mat[3] + mat[5];
            if (toggleButton.isChecked()){
                loc = adjustEstimation(loc);
            }
            drawExactResult(loc);
//                drawClassifiedResult(loc, toggleButton.isChecked());
//            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    private void drawClassifiedResult(float[] estimateGaze, boolean isShowDot){
        if( estimateGaze!=null ){
            switch (currentClassNum){
                case 4:
                    drawHandler.draw4ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result, isShowDot); break;
                case 6:
                    drawHandler.draw6ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result, isShowDot); break;
                case 9:
                    drawHandler.draw9ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result,  isShowDot); break;
            }
        }
    }

    private void drawExactResult(float[] estimateGaze){
        int landscapeHori = (int)(SCREEN_SIZE[0] * estimateGaze[0]);
        int landscapeVert = (int)(SCREEN_SIZE[1] * estimateGaze[1]);
        drawHandler.fillRect(landscapeHori, landscapeVert, 80,80, frame_gaze_result, R.color.estimated_square_color, false);
    }

    private float[] adjustEstimation(float[] newPoint){
        float[] average = new float[2];
        estimationList.add(new Point(newPoint[0], newPoint[1]));
        for (Point each : estimationList){
            average[0] += each.x;
            average[1] += each.y;
        }
        average[0] = average[0] / estimationList.size();
        average[1] = average[1] / estimationList.size();
        return average;
    }



}
