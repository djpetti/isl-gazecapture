package com.iai.mdf.Activities.Game;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.Dialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;
import android.util.Pair;
import android.view.Gravity;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.akexorcist.roundcornerprogressbar.RoundCornerProgressBar;
import com.iai.mdf.Activities.MainActivity;
import com.iai.mdf.DependenceClasses.DeviceConfiguration;
import com.iai.mdf.DependenceClasses.GameGrid;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.Handlers.SocketHandler;
import com.iai.mdf.R;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

/**
 * Created by mou on 3/23/18.
 */

public class GameMainActivity extends Activity {

    private final String LOG_TAG = "GameMainActivity";
    public static final String  KEY_GRID_SIZE = "grid_size";
    public static final String  KEY_AUTO_TRIGGER_THRESHOLD = "auto_trigger_common_sequence_size";
    public static final String  KEY_SPEED = "speed";
    public static final String  KEY_MODE = "game_mode";
    public static final int     VALUE_MODE_TIMER = 1;
    public static final String  KEY_MAX_SCORE = "game_max_score";
    public static final String  KEY_TRIGGER_MODE = "trigger_mode";
    public static final String  KEY_ADDITIONAL_VISUAL = "additional_visual_effect_circle";
    private static final int    GAME_DURATION = 1000 * 60;

    private SharedPreferences settings;
    private int GRID_SIZE;
    private int GAME_SPEED;
    private int GAME_AUTO_TRIGGER_THRESHOLD;
    private int GAME_MODE;
    private int GAME_MAX_SCORE;
    private boolean GAME_GAZE_AUTO_TRIGGER;
    private boolean GAME_ADDITIONAL_VISUAL;


    private ImageView       imgServerConnect;
    private ImageButton     btnController;
    private ImageButton     btnHammer;
    private ImageButton     btnExit;
    private View            table;
    private LinearLayout    tableRow1;
    private LinearLayout    tableRow2;
    private LinearLayout    tableRow3;
    private TextView        txtScore;
    private RoundCornerProgressBar progressBarTimer;
    private GameGrid        gameHandler;
    private CountDownTimer  gameCounterDownTimer;
    private GameResultDialog gameResultDialog;

    private int         curScore;

    private boolean     isTakingPicture = false;
    private boolean     isGameStarted = false;
    private boolean     isPreviewMode = false;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_game_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        hideBottomBar();


        initOpenCV();
        loadSettings();
        initGameGrid();
        initGameTimer();


        // score board
        txtScore = findViewById(R.id.activity_game_main_txt_score);
        txtScore.setText(String.valueOf(GAME_MAX_SCORE));
        curScore = 0;
        // image server connection indicator
        imgServerConnect = findViewById(R.id.activity_game_main_img_server_connect);
        imgServerConnect.setImageResource(R.drawable.game_main_server_invalid);
        imgServerConnect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if( !serverConnector.isConnected() ) {
                    imgServerConnect.setImageResource(R.drawable.game_main_server_timeout);
                    showToast("Connecting to the server...");
                    serverConnector.socketCreate();
                } else if (!isGameStarted) {
                    // when connection is made and game is not started,
                    // switch between preview and game mode
                    isPreviewMode = !isPreviewMode;
                    if( isPreviewMode ){
                        textureView.bringToFront();
                        if (!isTakingPicture) {
                            isTakingPicture = true;
                            takeImageHandler.post(takeImageRunnable);
                        }
                        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                                table.getHeight(),
                                table.getWidth()
                        );
                        params.gravity = Gravity.CENTER;
                        textureView.setLayoutParams(params);
                    } else {
                        table.bringToFront();
                        classifiedCircleHolder.bringToFront();
                    }
                }
            }
        });
        // controller button
        btnController = findViewById(R.id.activity_game_main_imgbtn_controller);
        btnController.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isTakingPicture) {
                    isTakingPicture = true;
                    takeImageHandler.post(takeImageRunnable);
                }
                if( isPreviewMode ){
                    table.bringToFront();
                    classifiedCircleHolder.bringToFront();
                    isPreviewMode = !isPreviewMode;
                }
                if(isGameStarted){
                    stopGame();
                } else {
                    startGame();
                }
                isGameStarted = !isGameStarted;
            }
        });
        // Hammer:  gaze manual trigger
        btnHammer = findViewById(R.id.activity_game_main_imgbtn_hammer);
        btnHammer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if( isGameStarted && !GAME_GAZE_AUTO_TRIGGER ){
                    ImageButton lastHole = gameHandler.getHole(GAZE_POS_QUEUE.get(GAZE_POS_QUEUE.size()-1));
                    if (lastHole!=null){
                        lastHole.performClick();
                    }
                }
            }
        });
        if( GAME_GAZE_AUTO_TRIGGER ){
            btnHammer.setVisibility(View.INVISIBLE);
        }
        // exit button
        btnExit = findViewById(R.id.activity_game_main_imgbtn_exit);
        btnExit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTakingPicture) {
                    isTakingPicture = false;
                    takeImageHandler.removeCallbacks(takeImageRunnable);
                }
                if( !isGameStarted ) {
                    finish();
                }
            }
        });


        // texture
        textureView = findViewById(R.id.activity_game_main_textureview);
        textureView.setRotation((float) 270.0);

        // circle holder
        classifiedCircleHolder = findViewById(R.id.activity_game_main_layout_circle_holder);
        classifiedCircleHolder.bringToFront();


        // prepare to start
        initGame();
        connectServer();



        frame_gaze_result = findViewById(R.id.activity_game_layout_dotHolder_result);
        frame_gaze_result.bringToFront();
        drawHandler = new DrawHandler(this, new int[]{confHandler.getScreenResoPWidth(),confHandler.getScreenResoPHeight()});

    }


    @Override
    protected void onResume() {
        super.onResume();
        btnController.setImageResource(R.drawable.button_main_start);   //can't be initialized in onCreate() or in layout
        connectServer();
        cameraInit();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(serverConnector!=null){
            serverConnector.socketDestroy();
        }
        cameraHandler.stopPreview();
        btnController.setImageResource(R.drawable.button_main_start);
        if( GAME_MODE== VALUE_MODE_TIMER) {
            gameCounterDownTimer.cancel();
        }
        gameHandler.stopGame();
        takeImageHandler.removeCallbacks(takeImageRunnable);
        isGameStarted = false;
    }

    @Override
    public void onBackPressed() {
        if( !isGameStarted ){
            super.onBackPressed();
        }
    }

    private void loadSettings(){
        Bundle extras = getIntent().getExtras();
        GRID_SIZE = extras.getInt(this.KEY_GRID_SIZE, 34);
        GAME_MODE = extras.getInt(this.KEY_MODE, this.VALUE_MODE_TIMER);
        settings = getSharedPreferences(MainActivity.PREFERENCE_NAME, Context.MODE_PRIVATE);
        GAME_SPEED = settings.getInt(this.KEY_SPEED, 5);
        GAME_AUTO_TRIGGER_THRESHOLD = settings.getInt(this.KEY_AUTO_TRIGGER_THRESHOLD, 4);
        GAME_MAX_SCORE = settings.getInt(this.KEY_MAX_SCORE, 0);
        GAME_GAZE_AUTO_TRIGGER = settings.getBoolean(this.KEY_TRIGGER_MODE, true);
        GAME_ADDITIONAL_VISUAL = settings.getBoolean(this.KEY_ADDITIONAL_VISUAL, false);
    }

    private void initGameGrid(){
        int rowNum = GRID_SIZE / 10;
        int colNum = GRID_SIZE % 10;
        gameHandler = new GameGrid(this, rowNum, colNum, GAME_SPEED);
        View.OnClickListener clickListener = new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if( isGameStarted ) {
                    Log.d(LOG_TAG, v.getTag().toString());
                    int point = gameHandler.whack((ImageButton)v);
                    curScore += point;
                    txtScore.setText(String.valueOf(curScore));
                }
            }
        };
        table     = findViewById(R.id.activity_game_main_layout_table);
        table.bringToFront();
        tableRow1 = findViewById(R.id.activity_game_main_table_row1);
        tableRow2 = findViewById(R.id.activity_game_main_table_row2);
        tableRow3 = findViewById(R.id.activity_game_main_table_row3);
        LinearLayout[] rows = new LinearLayout[] {tableRow1, tableRow2, tableRow3};
        LinearLayout.LayoutParams rowParam = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT,
                (float) 1 / rowNum
        );
        LinearLayout.LayoutParams cellParam = new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.MATCH_PARENT,
                1/(float)colNum);
        RelativeLayout.LayoutParams cellChildrenParam = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.MATCH_PARENT,
                RelativeLayout.LayoutParams.MATCH_PARENT);
        cellChildrenParam.addRule(RelativeLayout.CENTER_IN_PARENT, RelativeLayout.TRUE);

        for(int rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
            rows[rowIdx].setLayoutParams(rowParam);
            for (int colIdx = 0; colIdx < colNum; ++colIdx) {
                // Creating a new RelativeLayout as a cell
                RelativeLayout cell = new RelativeLayout(this);
                // image button
                ImageButton imgBtn = new ImageButton(this);
                imgBtn.setImageResource(R.drawable.mole_frame0);
                imgBtn.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
                imgBtn.setOnClickListener(clickListener);
                imgBtn.setBackgroundColor(Color.TRANSPARENT);
                imgBtn.setTag(rowIdx*10 + colIdx);
                imgBtn.setId(rowIdx*10 + colIdx);
                cell.addView(imgBtn, cellChildrenParam);
                rows[rowIdx].addView(cell, cellParam);
                // a drawing panel over image button
                ImageView imgView = new ImageView(this);
                imgView.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
                imgView.setImageResource(android.R.color.transparent);
                imgView.setBackgroundColor(Color.TRANSPARENT);
                imgView.setTag(rowIdx*10 + colIdx);
                imgView.setId(rowIdx*10 + colIdx);
                imgView.setClickable(false);
                cell.addView(imgView, cellChildrenParam);
                gameHandler.addHole(imgBtn, imgView);
            }
        }
    }

    private void initGameTimer(){
        progressBarTimer = findViewById(R.id.activity_game_main_progressbar);
        gameCounterDownTimer = new CountDownTimer(GAME_DURATION, 500) {
            public void onTick(long millisUntilFinished) {
                progressBarTimer.setProgress((float) millisUntilFinished/GAME_DURATION);
            }

            public void onFinish() {
                btnController.setImageResource(R.drawable.game_main_start);
                showGameResult();
                stopGame();
                isGameStarted = false;
            }
        };
    }

    private void initGame(){
        btnController.setImageResource(R.drawable.game_main_start);
        curScore = 0;
        txtScore.setText(String.valueOf(GAME_MAX_SCORE));
        if( GAME_MODE== VALUE_MODE_TIMER) {
            progressBarTimer.setProgress(1);
        }
    }

    private void startGame(){
        btnController.setImageResource(R.drawable.button_main_stop);
        curScore = 0;
        txtScore.setText(String.valueOf(curScore));
        if( GAME_MODE== VALUE_MODE_TIMER) {
            progressBarTimer.setProgress(1);
        }
        gameCounterDownTimer.start();
        gameHandler.startGame();
        if (!isTakingPicture) {
            isTakingPicture = true;
            takeImageHandler.post(takeImageRunnable);
        }
    }

    private void stopGame(){
        btnController.setImageResource(R.drawable.button_main_start);
        if( GAME_MODE== VALUE_MODE_TIMER) {
            gameCounterDownTimer.cancel();
        }
        gameHandler.stopGame();
        showGameResult();
        takeImageHandler.removeCallbacks(takeImageRunnable);
        drawHandler.clear(classifiedCircleHolder);
        isTakingPicture = false;
    }

    private void showGameResult(){
        final GameResultDialog dialog = new GameResultDialog(this, curScore);
        dialog.setCancelable(false);
        dialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        dialog.show();
        final Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (dialog.isShowing() ){
                    dialog.dismiss();
                }
            }
        }, 2500);
    }


    /****  Server ****/
    private SocketHandler serverConnector;
    private Toast toast;
    private double[]    optimalGaze = new double[]{1000,0,0};
    private int    optimalGazePosition = -1;

    private void connectServer(){
        SharedPreferences settings = getSharedPreferences(MainActivity.PREFERENCE_NAME, Context.MODE_PRIVATE);
        String lastIp = settings.getString(GameSettingActivity.BUNDLE_KEY_IP, null);
        String lastPort = settings.getString(GameSettingActivity.BUNDLE_KEY_PORT, "0");
        serverConnector = new SocketHandler(lastIp, Integer.parseInt(lastPort));
        serverConnector.setUiThreadHandler(new SocketHandler.StringCallback() {
            @Override
            public void onResponse(String str) {
                imgServerConnect.setImageResource(R.drawable.game_main_server_valid);
                try {
                    if (str.equalsIgnoreCase(SocketHandler.SUCCESS_CONNECT_MSG)){
                        if (isGameStarted) {
                            // if the connection comes back during the game
                            takeImageHandler.post(takeImageRunnable);
                        }
                        return;
                    }
                    if( str!=null ) {
                        JSONObject object = new JSONObject(str);
                        if (object != null ){
                            if (object.getBoolean("Valid")) {
                                optimalGazePosition = analyzeGaze(object);
                                clickTriggle(optimalGazePosition);
                                Log.d(LOG_TAG, object.toString());
                            } else {
                                Log.d(LOG_TAG, "inValid");
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
                if( str.equalsIgnoreCase(GameServerConnector.ERROR_DISCONNECTED) ){
                    imgServerConnect.setImageResource(R.drawable.game_main_server_invalid);
                    takeImageHandler.removeCallbacks(takeImageRunnable);
                    isTakingPicture = false;
//                } else if (str.equalsIgnoreCase(GameServerConnector.ERROR_TIMEOUT)) {
//                    imgServerConnect.setImageResource(R.drawable.game_main_server_timeout);
                } else if (str.equalsIgnoreCase(SocketHandler.ERROR_SETTING)) {
                    Toast.makeText(GameMainActivity.this, "Please check the address and the port to use eye controller", Toast.LENGTH_SHORT).show();
                    imgServerConnect.setImageResource(R.drawable.game_main_server_invalid);
                }
            }
        });
        serverConnector.socketCreate();
    }

    public void showToast(final String msg){
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                toast = Toast.makeText(GameMainActivity.this, msg, Toast.LENGTH_SHORT);
                toast.show();
            }
        });
    }



    /****  Camera ****/
    private BaseLoaderCallback openCVLoaderCallback;
    private CameraHandler   cameraHandler;
    private DeviceConfiguration confHandler = DeviceConfiguration.getInstance(this);
    private TextureView     textureView;
    private Handler         takeImageHandler = new Handler();
    private Runnable        takeImageRunnable;
    private Handler         uiThreadHandler = null;
    private int             prevReceivedGazeIndex = 0;
    private ArrayList<Point> estimationList = new ArrayList<>();


    private void cameraInit(){
        cameraHandler = new CameraHandler(this, true);
        cameraHandler.setOnImageAvailableListenerForPrev(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                Image image = reader.acquireNextImage();
                if( cameraHandler.getCameraState()==CameraHandler.CAMERA_STATE_STILL_CAPTURE ) {
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    Log.d(LOG_TAG, "Take a picture");
                    serverConnector.uploadImage(image, confHandler);
                }
                image.close();
            }
        });
        cameraHandler.startPreview(textureView);
        takeImageRunnable = new Runnable() {
            @Override
            public void run() {
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
                takeImageHandler.postDelayed(this, confHandler.getDemoCaptureDelayTime());
            }
        };
    }


    private void initOpenCV(){
        // used when loading openCV4Android
        openCVLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        Log.d(LOG_TAG, "OpenCV loaded successfully");
                        break;
                    default:
                        super.onManagerConnected(status);
                    break;
                }
            }
        };
        if (!OpenCVLoader.initDebug()) {
            Log.d(LOG_TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, openCVLoaderCallback);
        } else {
            Log.d(LOG_TAG, "OpenCV library found inside package. Using it!");
            openCVLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    /***** Gaze Control *****/
//    private ArrayList<Pair<Float,Float>> pointHistory = new ArrayList<>();
    private LinkedList<Pair<Float,Float>> pointHistory = new LinkedList<>();
    private ArrayList<Integer> GAZE_POS_QUEUE = new ArrayList<>();
    private final double     CLICK_THRESHOLD = 0.1;
    private final int       NUM_OF_PRE_POINTS = 1;

    private int analyzeGaze(JSONObject object){
        try {
            double portraitHori = object.getDouble(SocketHandler.JSON_KEY_PREDICT_Y);
            double portraitVert = object.getDouble(SocketHandler.JSON_KEY_PREDICT_X);
            float[] loc = new float[2];
            loc[1] = 1 - (float)((portraitHori + confHandler.getCameraOffsetPWidth())/confHandler.getScreenSizePWidth());
            loc[0] = (float)((portraitVert + confHandler.getCameraOffsetPHeight())/confHandler.getScreenSizePHeight());
            // linear calibration
            float[] mat = confHandler.getCalibrationMatrix();
            loc[0] = loc[0] * mat[0]  + loc[1] * mat[2] + mat[4];
            loc[1] = loc[0] * mat[1]  + loc[1] * mat[3] + mat[5];
            pointHistory.add(new Pair(loc[0], loc[1]));
            if (pointHistory.size()>NUM_OF_PRE_POINTS) {
                pointHistory.remove(0);
            }
            // analyze last three points distribution
            double avePX = 0;
            double avePY = 0;
            drawHandler.clear(frame_gaze_result);
            for(Pair<Float, Float> eachPoint: pointHistory){
                avePX += eachPoint.first;
                avePY += eachPoint.second;
//                drawExactResult( new float[]{eachPoint.first, eachPoint.second}, true, R.color.estimated_square_color);
            }
            avePX /= pointHistory.size();
            avePY /= pointHistory.size();
            int gridPos = -1;
            if( isPreviewMode ) {
                drawExactResult(new float[]{(float) avePX, (float) avePY}, true, R.color.desired_square_color);
            } else if( GAME_ADDITIONAL_VISUAL ){
                gridPos = drawClassifiedResult(new float[]{(float) avePX, (float) avePY});
            }
            Log.d(LOG_TAG, String.valueOf(gridPos));
            return gridPos;
//            double diff = 0;
//            for(Pair<Float, Float> eachPoint : pointHistory){
//                diff += Math.sqrt((eachPoint.first - avePX)*(eachPoint.first - avePX)
//                        + (eachPoint.second - avePY)*(eachPoint.second - avePY));
//            }
//            diff /= pointHistory.size();
//            Log.d(LOG_TAG, String.valueOf(diff));
//            return new double[]{diff, avePX, avePY};
        } catch (JSONException e) {
            e.printStackTrace();
            return -1;
        }
    }

    private void clickTriggle(double[] analRes){
        if (analRes==null){
            return;
        }
        double diff = analRes[0];
        double avePX = analRes[1];
        double avePY = analRes[2];
        if(diff < CLICK_THRESHOLD
                && avePX > 0 && avePY > 0
                && avePX < (double)table.getHeight()/confHandler.getScreenResoPWidth()
                && avePY < (double)table.getWidth()/confHandler.getScreenResoPHeight()){
            Log.d(LOG_TAG, "Click");
            int rowNum = GRID_SIZE / 10;
            int colNum = GRID_SIZE % 10;
            int btnRowIdx = rowNum - 1 - (int)(avePX / ((double)table.getHeight()/confHandler.getScreenResoPWidth()/rowNum));
            int btnColIdx = (int)(avePY / ((double)table.getWidth()/confHandler.getScreenResoPHeight()/colNum));
            if ( btnRowIdx*colNum + btnColIdx < gameHandler.getHoles().size()) {
                gameHandler.getHole(btnRowIdx * colNum + btnColIdx).performClick();
            }
        } else {
            Log.d(LOG_TAG, "Moving");
        }
    }

    private void clickTriggle(int gazePosition){
        GAZE_POS_QUEUE.add(gazePosition);
        if (GAZE_POS_QUEUE.size() > GAME_AUTO_TRIGGER_THRESHOLD){
            GAZE_POS_QUEUE.remove(0);
        }
//        // percentage
//        Collections.sort(GAZE_POS_QUEUE);
//        int maxNum = -1, maxCount = 0, tempNum = -1, tempCount = 0;
//        for (int i =0; i < GAZE_POS_QUEUE.size(); ++i) {
//            if (GAZE_POS_QUEUE.get(i) == tempNum){
//                tempCount++;
//            } else {
//                if (tempCount > maxCount){
//                    maxCount = tempCount;
//                    maxNum = tempNum;
//                }
//                tempCount = 1;
//                tempNum = GAZE_POS_QUEUE.get(i);
//            }
//        }
//        if (maxCount > 3){
//
//        }
        if (!GAME_GAZE_AUTO_TRIGGER){
            return;
        }
        boolean theSame = true;
        for (int i=0; i < GAZE_POS_QUEUE.size()-1; ++i){
            if( GAZE_POS_QUEUE.get(i) != GAZE_POS_QUEUE.get(i+1)){
                theSame = false;
                break;
            }
        }
        if( theSame && GAZE_POS_QUEUE.size() >= 2){
            ImageButton gazedHole = gameHandler.getHole(0);
            if (gazedHole!=null){
                gazedHole.performClick();
            }
        } else {
            Log.d(LOG_TAG, "Moving");
        }
    }


    private int[] getMostFrequent(int arr[]) {
        // Insert all elements in hash
        Map<Integer, Integer> hp = new HashMap();
        for(int i = 0; i < arr.length; i++)
        {
            int key = arr[i];
            if(hp.containsKey(key))
            {
                int freq = hp.get(key);
                freq++;
                hp.put(key, freq);
            }
            else
            {
                hp.put(key, 1);
            }
        }
        // find max frequency.
        int max_count = 0, res = -1;
        for(Map.Entry<Integer, Integer> val : hp.entrySet())
        {
            if (max_count < val.getValue())
            {
                res = val.getKey();
                max_count = val.getValue();
            }
        }
        return new int[]{res, max_count};
    }


    /***** Drawing *****/
    private FrameLayout frame_gaze_result;
    private FrameLayout classifiedCircleHolder;
    private DrawHandler drawHandler;
//    private int[]       SCREEN_SIZE = new int[]{1440, 2392};
    private void drawGaze(JSONObject object, boolean isHoldOn, int color){
        try {
            double portraitHori = object.getDouble(SocketHandler.JSON_KEY_PREDICT_Y);
            double portraitVert = object.getDouble(SocketHandler.JSON_KEY_PREDICT_X);
            float[] loc = new float[2];
            loc[0] = (float)((portraitHori + confHandler.getCameraOffsetPWidth())/confHandler.getScreenSizePWidth());
            loc[1] = (float)((portraitVert + confHandler.getCameraOffsetPHeight())/confHandler.getScreenSizePHeight());
            drawExactResult(loc, isHoldOn, color);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    private void drawExactResult(float[] estimateGaze, boolean isHoldOn, int color){
        int portraitX = (int)(confHandler.getScreenResoPHeight() * estimateGaze[0]);
        int portraitY = (int)(confHandler.getScreenResoPWidth() * estimateGaze[1]);
        drawHandler.fillRect(portraitX, portraitY, 80,80, frame_gaze_result, color, isHoldOn);
    }

    private int drawClassifiedResult(float[] estimateGaze){
        int holderWidth     = classifiedCircleHolder.getWidth();
        int holderHeight    = classifiedCircleHolder.getHeight();
        float gazeToLeft    = confHandler.getScreenResoPHeight() * estimateGaze[0];
        float gazeToBottom  = confHandler.getScreenResoPWidth() * (1 - estimateGaze[1]);
        float[] relPos = new float[2];
        relPos[0] = gazeToLeft / holderWidth;
        relPos[1] = 1 - gazeToBottom / holderHeight;
        int gridPos = -1;
        if( estimateGaze!=null ){
            switch (GRID_SIZE){
                case 22:
                    gridPos = drawHandler.draw22ClassifiedResult(relPos, new int[]{holderWidth, holderHeight}, classifiedCircleHolder); break;
                case 33:
                    gridPos = drawHandler.draw33ClassifiedResult(relPos, new int[]{holderWidth, holderHeight}, classifiedCircleHolder); break;
                case 34:
                    gridPos = drawHandler.draw34ClassifiedResult(relPos, new int[]{holderWidth, holderHeight}, classifiedCircleHolder); break;
            }
        } else {
            classifiedCircleHolder.removeAllViews();
        }
        return gridPos;
    }



    /****  Window Setting ****/

    @SuppressLint("NewApi")
    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if(android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT && hasFocus) {
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
        }
    }


    private void hideBottomBar(){
        final int flags = View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
        // This work only for android 4.4+
        if(android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            getWindow().getDecorView().setSystemUiVisibility(flags);
            final View decorView = getWindow().getDecorView();
            decorView.setOnSystemUiVisibilityChangeListener(new View.OnSystemUiVisibilityChangeListener() {
                @Override
                public void onSystemUiVisibilityChange(int visibility) {
                    if((visibility & View.SYSTEM_UI_FLAG_FULLSCREEN) == 0) {
                        decorView.setSystemUiVisibility(flags);
                    }
                }
            });
        } else {
            View v = this.getWindow().getDecorView();
            v.setSystemUiVisibility(View.GONE);
        }
    }






    private class GameResultDialog extends Dialog {

        private TextView    textView;
        private int         score;

        public GameResultDialog(@NonNull Context context, int s) {
            super(context);
            score = s;
        }


        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            requestWindowFeature(Window.FEATURE_NO_TITLE); //before
            setContentView(R.layout.dialog_game);
            textView = findViewById(R.id.dialog_game_txt_result);
            String str = "Game Over! \n" + String.valueOf(curScore);
            if( score > GAME_MAX_SCORE){
                str += "\nBest Score!";
                SharedPreferences.Editor editor = settings.edit();
                editor.putInt(KEY_MAX_SCORE, curScore);
                editor.commit();
            }
            textView.setText(str);
        }
    }



}
