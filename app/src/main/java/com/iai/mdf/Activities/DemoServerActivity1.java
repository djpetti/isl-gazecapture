package com.iai.mdf.Activities;

import android.content.pm.ActivityInfo;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.iai.mdf.DependenceClasses.Configuration;
import com.iai.mdf.DependenceClasses.DeviceProfile;
import com.iai.mdf.FaceDetectionAPI;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.Handlers.ImageProcessHandler;
import com.iai.mdf.Handlers.SocketHandler;
import com.iai.mdf.Handlers.TensorFlowHandler;
import com.iai.mdf.Handlers.TimerHandler;
import com.iai.mdf.R;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

//import com.moutaigua.isl_android_gaze.FaceDetectionAPI;

/**
 * Created by Mou on 9/22/2017.
 */

public class DemoServerActivity1 extends AppCompatActivity {

    public static final String BUNDLE_KEY_IP = "ip";
    public static final String BUNDLE_KEY_PORT = "port";
    private final String LOG_TAG = "DemoServerActivity2";
    private final String JSON_STRING_START = "RESP_START";
    private final String JSON_KEY_PREDICT_X = "PredictX";
    private final String JSON_KEY_PREDICT_Y = "PredictY";
    private final String JSON_KEY_SEQ_NUMBER = "SequenceNumber";


    private CameraHandler cameraHandler;
    private DrawHandler drawHandler;
    private TextureView textureView;
    private Spinner     spinnerView;
    private ToggleButton toggleButton;
    private FrameLayout frame_background_grid;
    private FrameLayout view_dot_container;
    private FrameLayout frame_gaze_result;
    private FrameLayout frame_bounding_box;
    private TextView    result_board;
    private int[]       SCREEN_SIZE;
    private int[]       TEXTURE_SIZE;
    private BaseLoaderCallback openCVLoaderCallback;
    private boolean isRealTimeDetection = false;
    private Handler autoDetectionHandler = new Handler();
    private Runnable autoDetectionRunnable;
    private TensorFlowHandler tensorFlowHandler;
    private int         mFrameIndex = 0;
    private int         currentClassNum = 4;
    private SocketHandler socketHandler;
    private int         prevReceivedGazeIndex = 0;
    private String  socketIp = null;
    private int     socketPort;
    private Configuration confHandler = Configuration.getInstance(this);



    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_demo);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        getSupportActionBar().hide();
        Bundle extras = getIntent().getExtras();
        socketIp = extras.getString(BUNDLE_KEY_IP);
        socketPort = extras.getInt(BUNDLE_KEY_PORT);

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


        view_dot_container = findViewById(R.id.activity_demo_layout_dotHolder_background);
        frame_background_grid = findViewById(R.id.activity_demo_layout_background_grid);
        frame_background_grid.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));
//        frame_background_grid.bringToFront();
        frame_gaze_result = findViewById(R.id.activity_demo_layout_dotHolder_result);
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
                isRealTimeDetection = !isRealTimeDetection;
                if(isRealTimeDetection){
                    frame_background_grid.setBackgroundColor(0xFFFFFFFF);   // cover texture with white
                    autoDetectionHandler.post(autoDetectionRunnable);
                    result_board.setText("");
                } else {
                    frame_background_grid.setBackgroundColor(0x00FFFFFF);   // uncover texture with translucent
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    autoDetectionHandler.removeCallbacks(autoDetectionRunnable);
                    result_board.setText("Press Anywhere to Start");
                    Log.d(LOG_TAG, "Ave Time: " + String.valueOf(total_time/mFrameIndex) + ";   Frames = " + String.valueOf(mFrameIndex));
                }
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
                            switchBackground(frame_background_grid, R.layout.grid4_for_demo);
                            Log.d(LOG_TAG, "Selected: 2x2");
                            break;
                        case "2x3":
                            currentClassNum = 6;
                            switchBackground(frame_background_grid, R.layout.grid6_for_demo);
                            Log.d(LOG_TAG, "Selected: 2x3");
                            break;
                        case "3x3":
                            currentClassNum = 9;
                            switchBackground(frame_background_grid, R.layout.grid9_for_demo);
                            Log.d(LOG_TAG, "Selected: 3x3");
                            break;
                    }
                } else {
                    // return to the previous state
                    switch (currentClassNum){
                        case 4:
                            spinnerView.setSelection(0);
                            break;
                        case 6:
                            spinnerView.setSelection(1);
                            break;
                        case 9:
                            spinnerView.setSelection(2);
                            break;
                    }

                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
                Log.d(LOG_TAG, "Selected Nothing ");
            }
        });
        spinnerView.bringToFront();

        toggleButton = findViewById(R.id.activity_demo_toggle_show_dot);
        toggleButton.setChecked(false);

        result_board = findViewById(R.id.activity_demo_txtview_result);
        result_board.setText("Press Anywhere to Start");

        autoDetectionRunnable = new Runnable() {
            @Override
            public void run() {
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
//                drawHandler.clear(frame_bounding_box);
                drawHandler.clear(frame_gaze_result);
                autoDetectionHandler.postDelayed(this, confHandler.getDemoCaptureDelayTime());
            }
        };


        tensorFlowHandler = new TensorFlowHandler(this);
        tensorFlowHandler.pickModel(TensorFlowHandler.MODEL_ISL_FILE_NAME);


        // load device profile
//        ArrayList<DeviceProfile> allDevices = DeviceProfile.loadDeviceProfileList(this);
//        deviceProfile = DeviceProfile.getProfileByName(allDevices, Build.MODEL);
//        Log.d(LOG_TAG, String.valueOf(deviceProfile.getCollectionCaptureDelayTime()));

    }


    @Override
    public void onResume() {
        super.onResume();
        cameraHandler = CameraHandler.getInstance(this, true);
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
                        uploadImageOnBLU(image);
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
        autoDetectionHandler.removeCallbacks(autoDetectionRunnable);
        isRealTimeDetection = false;
        // stop socket communication
        if(socketHandler!=null) {
            socketHandler.socketDestroy();
        }
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
        socketHandler.setConnectCallback(new SocketHandler.StringCallback() {
            @Override
            public void onResponse(String str) {
            }

            @Override
            public void onError(String str) {
                showToast("Please set the address and the port correctly");
            }
        });
        socketHandler.socketCreate();
        socketHandler.setUiThreadHandler(new SocketHandler.StringCallback() {
            @Override
            public void onResponse(String str) {
                try {
                    if( str!=null ) {
                        JSONObject object = new JSONObject(str);
                        if (object != null) {
                            Log.d(LOG_TAG, object.toString());
                            if( object.getBoolean("Valid") && isRealTimeDetection ) {
                                drawGaze(object);
                                Log.d(LOG_TAG, "draw");
                            } else {
                                showToast("No Detection");
                                Log.d(LOG_TAG, "no detection");
                            }
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
                    showToast("Disconnect From Server\nRestart Please");
                    frame_bounding_box.performClick();
                } else if (str.equalsIgnoreCase(SocketHandler.ERROR_TIMEOUT)) {
                    showToast("Timeout");
                }
            }
        });
    }


    private long total_time = 0;
    private void uploadImageOnBLU(Image image){
        TimerHandler.getInstance().tic();
        Mat colorImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC4);
        ImageProcessHandler.getRGBMat(image, colorImg.getNativeObjAddr());
        Imgproc.cvtColor(colorImg, colorImg, Imgproc.COLOR_BGRA2BGR);
//
        byte[] jpegBytes = ImageProcessHandler.fromMatToJpegByte(colorImg);
//        byte[] imageBytes = ImageProcessHandler.fromMatToJpegByte2(colorImg);
        Log.d(LOG_TAG, "Image Format Conversion: " + String.valueOf(TimerHandler.getInstance().toc()));
        total_time += TimerHandler.getInstance().toc();
        byte[] sizeBytes = ByteBuffer.allocate(4).putInt(jpegBytes.length).order(ByteOrder.nativeOrder()).array();
        byte[] seqBytes = new byte[2];
        seqBytes[0] = (byte)(mFrameIndex & 0xFF);
        byte[] data = new byte[jpegBytes.length + 5];
        System.arraycopy(sizeBytes, 0, data, 0, 4);
        System.arraycopy(jpegBytes, 0, data, 4, jpegBytes.length);
        System.arraycopy(seqBytes, 0, data, 4 + jpegBytes.length, 1);
        socketHandler.send(data);
        mFrameIndex++;
    }









    private Toast toast;
    public void showToast(final String msg){
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                toast = Toast.makeText(DemoServerActivity1.this, msg, Toast.LENGTH_SHORT);
                toast.show();
            }
        });
    }

    private int[] fetchScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }

    private void switchBackground(FrameLayout layoutHolder, int layoutId){
        layoutHolder.removeAllViews();
        LayoutInflater inflater = (LayoutInflater) this.getSystemService(LAYOUT_INFLATER_SERVICE);
        View childLayout = inflater.inflate(layoutId, (ViewGroup) findViewById(R.id.grid_for_demo));
        layoutHolder.addView(childLayout);
    }

    private void drawGaze(JSONObject object){
        try {
            int receivedIdx = object.getInt(JSON_KEY_SEQ_NUMBER);
            if( receivedIdx > prevReceivedGazeIndex ){
                prevReceivedGazeIndex = receivedIdx;
                double portraitHori = object.getDouble(JSON_KEY_PREDICT_Y);
                double portraitVert = object.getDouble(JSON_KEY_PREDICT_X);
                float[] loc = new float[2];
//                loc[0] = (float)((portraitHori + deviceProfile.getCameraOffsetX())/deviceProfile.getScreenSizeX());
//                loc[1] = (float)((portraitVert + deviceProfile.getCameraOffsetY())/deviceProfile.getScreenSizeY());
                loc[0] = (float) (portraitHori + confHandler.getCameraOffsetX())/confHandler.getScreenSizeX();
                loc[1] = (float) (portraitVert + confHandler.getCameraOffsetY())/confHandler.getScreenSizeY();
                drawClassifiedResult(loc, toggleButton.isChecked());
            }
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





}
