package com.iai.mdf.Activities;

import android.content.pm.ActivityInfo;
import android.graphics.Point;
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
import android.view.LayoutInflater;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.iai.mdf.DependenceClasses.DeviceConfiguration;
import com.iai.mdf.DependenceClasses.Matrix;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.Handlers.SocketHandler;
import com.iai.mdf.Handlers.TensorFlowHandler;
import com.iai.mdf.Handlers.TextFileHanlder;
import com.iai.mdf.R;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.util.ArrayList;

//import com.moutaigua.isl_android_gaze.FaceDetectionAPI;

/**
 * Created by Mou on 9/22/2017.
 */

public class CalibrationActivity extends AppCompatActivity {

    public static final String BUNDLE_KEY_IP = "ip";
    public static final String BUNDLE_KEY_PORT = "port";
    private final String LOG_TAG = "CalibrationActivity";
    private final String JSON_STRING_START = "RESP_START";


    private CameraHandler cameraHandler;
    private DrawHandler drawHandler;
    private TextureView textureView;
    private FrameLayout textureviewHolder;
    private FrameLayout view_dot_container;
    private TextView    result_board;
    private int[]       SCREEN_SIZE;
    private int[]       TEXTURE_SIZE;
    private BaseLoaderCallback openCVLoaderCallback;
    private boolean     isRealTimeDetection = false;
    private Handler     autoDetectionHandler = new Handler();
    private Runnable    takePicRunnable;
    private Handler     dotGeneratorHandler = new Handler();
    private Runnable    dotGeneratorRunnable;
    private Handler     matrixComputationHandler = new Handler();
    private Runnable    matrixComputationRunnable;
    private TensorFlowHandler tensorFlowHandler;
    private int         mFrameIndex = 0;
    private int         currentClassNum = 4;
    private SocketHandler socketHandler;
    private int         prevReceivedGazeIndex = 0;
    private String      socketIp = null;
    private int         socketPort;
    private DeviceConfiguration confHandler = DeviceConfiguration.getInstance(this);
    /** calibration **/
    private int                     AmountPicForEachPoint = 2;
    private ArrayList<float[]>      GroundTruthPoints = new ArrayList<>();
    private ArrayList<float[]>      EstimatePoints = new ArrayList<>();
    private int  temp_counter = 0;
    private long    timestamp = 0;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_calibration);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getSupportActionBar().hide();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        Bundle extras = getIntent().getExtras();
        socketIp = extras.getString(BUNDLE_KEY_IP);
        socketPort = extras.getInt(BUNDLE_KEY_PORT);

        // init openCV
        initOpenCV();

        SCREEN_SIZE = fetchScreenSize();
        textureView = findViewById(R.id.activity_calibration_preview_textureview);
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
        textureviewHolder = findViewById(R.id.activity_calibration_layout_textureview_holder);
        textureviewHolder.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                TEXTURE_SIZE[1], TEXTURE_SIZE[0]
        );
        params.gravity = Gravity.CENTER;
        textureView.setLayoutParams(params);
//        textureView.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));

        result_board = findViewById(R.id.activity_calibration_txtview_result);
        result_board.setText("Press Anywhere to Start");

        view_dot_container = findViewById(R.id.activity_calibration_layout_dotHolder_background);
        view_dot_container.bringToFront();
        view_dot_container.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawHandler.clear(view_dot_container);
                Log.d(LOG_TAG, "pressed");  //cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
                if( !socketHandler.isConnected() && !isRealTimeDetection ){
                    Toast.makeText(CalibrationActivity.this, "Restart to connect to the server", Toast.LENGTH_SHORT).show();
                    return;
                }
                isRealTimeDetection = !isRealTimeDetection;
                if(isRealTimeDetection){
                    EstimatePoints.clear();
                    GroundTruthPoints.clear();
                    view_dot_container.setBackgroundColor(0xFFFFFFFF);   // cover texture with white
                    dotGeneratorHandler.postDelayed(dotGeneratorRunnable, 800);
//                    autoDetectionHandler.postDelayed(takePicRunnable, 500+confHandler.getCalibrationSpeed()/AmountPicForEachPoint/2);
                    result_board.setText("");
                } else {
                    view_dot_container.setBackgroundColor(0x00FFFFFF);   // uncover texture with translucent
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    dotGeneratorHandler.removeCallbacks(dotGeneratorRunnable);
                    autoDetectionHandler.removeCallbacks(takePicRunnable);
                    // trunk to the same size
                    while (GroundTruthPoints.size() != EstimatePoints.size()){
                        GroundTruthPoints.remove(GroundTruthPoints.size()-1);
                    }
                    // delete invalid frames first in the list
                    for (int i=0; i < EstimatePoints.size(); i++) {
                        if (Math.abs(EstimatePoints.get(i)[0] + 1) < 0.0000001 && Math.abs(EstimatePoints.get(i)[1] + 1) < 0.0000001) {
                            GroundTruthPoints.remove(i);
                            EstimatePoints.remove(i);
                            i--;
                        }
                    }
                    Toast.makeText(CalibrationActivity.this, "Sample Collected: " + String.valueOf(GroundTruthPoints.size()), Toast.LENGTH_LONG).show();
                    if (GroundTruthPoints.size()<2){
                        result_board.setText("Too few samples are collected\nPress Anywhere to Restart");
                    } else {
                        float[] mat = computeTransportationMaxtrix(EstimatePoints, GroundTruthPoints);
                        confHandler.setCalibrationMatrix(mat);
                        confHandler.saveConfiguration();
                        Log.d(LOG_TAG, String.valueOf(mat[0]));
                        Log.d(LOG_TAG, String.valueOf(mat[1]));
                        Log.d(LOG_TAG, String.valueOf(mat[2]));
                        Log.d(LOG_TAG, String.valueOf(mat[3]));
                        Log.d(LOG_TAG, String.valueOf(mat[4]));
                        Log.d(LOG_TAG, String.valueOf(mat[5]));
                        result_board.setText("Calibration is done: \n" +
                                            String.valueOf(mat[0]).substring(0, 6) + ", " + String.valueOf(mat[1]).substring(0, 6) + ";\n" +
                                            String.valueOf(mat[2]).substring(0, 6) + ", " + String.valueOf(mat[3]).substring(0, 6) + ";\n" +
                                            String.valueOf(mat[4]).substring(0, 6) + ", " + String.valueOf(mat[5]).substring(0, 6) + ";");
                    }
                }
            }
        });

        drawHandler = new DrawHandler(this, fetchScreenSize());
        drawHandler.setDotHolderLayout(view_dot_container);
        drawHandler.showAllCandidateDots();

        dotGeneratorRunnable = new Runnable() {
            @Override
            public void run() {
                drawHandler.clear(view_dot_container);
                drawHandler.showNextPointInOrder();
                autoDetectionHandler.removeCallbacks(takePicRunnable);
                temp_counter = 0;
                Point curPoint = drawHandler.getCurrDot();
                for(int i=0; i<AmountPicForEachPoint; i++) {
                    GroundTruthPoints.add(new float[]{(float) curPoint.x / (float) SCREEN_SIZE[0], (float) curPoint.y / (float) SCREEN_SIZE[1]});
                }
//                delayCapture(confHandler.getCollectionCaptureDelayTime());
                autoDetectionHandler.postDelayed(takePicRunnable, (long)(confHandler.getCalibrationSpeed()/(AmountPicForEachPoint+1)*1.5));
                dotGeneratorHandler.postDelayed(this,confHandler.getCalibrationSpeed());
            }
        };

        takePicRunnable = new Runnable() {
            @Override
            public void run() {
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
                temp_counter++;
                timestamp = System.currentTimeMillis();
                Log.d("aaaa", "image No." + String.valueOf(temp_counter));
                autoDetectionHandler.postDelayed(this, confHandler.getCalibrationSpeed()/(AmountPicForEachPoint+1));
            }
        };

//        tensorFlowHandler = new TensorFlowHandler(this);
//        tensorFlowHandler.pickModel(TensorFlowHandler.MODEL_ISL_FILE_NAME);

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
                    Log.d("aaaa", String.valueOf(System.currentTimeMillis() - timestamp));
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
                            double landscapeHori = object.getDouble(SocketHandler.JSON_KEY_PREDICT_X);
                            double landscapeVert = object.getDouble(SocketHandler.JSON_KEY_PREDICT_Y);
                            float[] loc = new float[2];
                            loc[0] = (float) (landscapeHori + confHandler.getCameraOffsetPHeight())/confHandler.getScreenSizePWidth();
                            loc[1] = 1 - (float) (landscapeVert + confHandler.getCameraOffsetPWidth())/confHandler.getScreenSizePHeight();
                            EstimatePoints.add(new float[]{ loc[0], loc[1] } );
                            Log.d(LOG_TAG, object.toString());
                        } else {
                            EstimatePoints.add(new float[]{ -1, -1 } );
                            Log.d(LOG_TAG, "invalid");
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
                    Toast.makeText(CalibrationActivity.this, "Disconnect From Server\nRestart Please", Toast.LENGTH_SHORT).show();
                    view_dot_container.performClick();
                } else if (str.equalsIgnoreCase(SocketHandler.ERROR_TIMEOUT)) {
                    Log.d(LOG_TAG, "Timeout");
                } else if (str.equalsIgnoreCase(SocketHandler.ERROR_SETTING)) {
                    Toast.makeText(CalibrationActivity.this, "Please set the address and the port correctly", Toast.LENGTH_SHORT).show();
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

    private void switchGridBackground(FrameLayout layoutHolder, int layoutId){
        layoutHolder.removeAllViews();
        LayoutInflater inflater = (LayoutInflater) this.getSystemService(LAYOUT_INFLATER_SERVICE);
        View childLayout = inflater.inflate(layoutId, (ViewGroup) findViewById(R.id.grid_for_demo));
        layoutHolder.addView(childLayout);
    }


    private void delayCapture(int delayLength){
        Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
            }
        }, delayLength);
    }


    private float[] computeTransportationMaxtrix(ArrayList<float[]> leftMat, ArrayList<float[]> rightMat){
        // leftMat * A = rightMat
        // Estimated * A = Ground Truth
        // form the matrices
        double[][] estArr = new double[leftMat.size()][3];
        double[][] truArr = new double[rightMat.size()][2];
        TextFileHanlder.WriteLogIntoFile(leftMat);
        TextFileHanlder.WriteLogIntoFile(rightMat);
//        TextFileHanlder.WriteLogIntoFile(leftMatValid);
//        TextFileHanlder.WriteLogIntoFile(rightMatValid);
        for ( int i=0; i < leftMat.size(); i++){
            estArr[i][0] = leftMat.get(i)[0];
            estArr[i][1] = leftMat.get(i)[1];
            estArr[i][2] = 1;
            truArr[i][0] = rightMat.get(i)[0];
            truArr[i][1] = rightMat.get(i)[1];
        }
        Log.d(LOG_TAG, "leftMat");
        for ( int i=0; i < leftMat.size(); i++){
            Log.d(LOG_TAG, "("+String.valueOf(leftMat.get(i)[0])+", "+String.valueOf(leftMat.get(i)[1])+")");
        }
        Log.d(LOG_TAG, "rightMat");
        for ( int i=0; i < rightMat.size(); i++){
            Log.d(LOG_TAG, "("+String.valueOf(rightMat.get(i)[0])+", "+String.valueOf(rightMat.get(i)[1])+")");
        }
        Matrix estMat = new Matrix(estArr);
        Matrix truMat = new Matrix(truArr);
        Matrix estMatT = estMat.transpose();
        Matrix temp1 = estMatT.times(estMat);   // x^T * x
        Matrix temp1_inv = temp1.inverse3x3();
        Matrix temp2 = estMatT.times(truMat);   // X^T * y
        Matrix res = temp1_inv.times(temp2);    // (x^T * x)^-1 * x^T * y
        return new float[]{
                (float) res.get(0,0),
                (float) res.get(0,1),
                (float) res.get(1,0),
                (float) res.get(1,1),
                (float) res.get(2,0),
                (float) res.get(2,1)};
    }


}
