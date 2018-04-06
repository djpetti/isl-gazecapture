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
import android.os.SystemClock;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;
import android.util.Pair;
import android.view.MotionEvent;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.akexorcist.roundcornerprogressbar.RoundCornerProgressBar;
import com.iai.mdf.Activities.MainActivity;
import com.iai.mdf.DependenceClasses.Configuration;
import com.iai.mdf.DependenceClasses.GameGrid;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.R;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;

import java.util.ArrayList;

/**
 * Created by mou on 3/23/18.
 */

public class GameMainActivity extends Activity {

    private final String LOG_TAG = "GameMainActivity";
    public static final String KEY_GRID_SIZE = "grid_size";
    public static final String KEY_SPEED = "speed";
    public static final String KEY_MODE = "game_mode";
    public static final int     VALUE_MODE_TIMER = 1;
    public static final String KEY_MAX_SCORE = "game_max_score";
    private static final int GAME_DURATION = 1000 * 60;

    private SharedPreferences settings;
    private int GRID_SIZE;
    private int GAME_SPEED;
    private int GAME_MODE;
    private int GAME_MAX_SCORE;


    private ImageView   imgServerConnect;
    private ImageButton btnController;
    private ImageButton btnExit;
    private View            table;
    private LinearLayout    tableRow1;
    private LinearLayout    tableRow2;
    private LinearLayout    tableRow3;
    private TextView    txtScore;
    private RoundCornerProgressBar progressBarTimer;
    private GameGrid    gameHandler;
    private CountDownTimer  gameCounterDownTimer;
    private GameResultDialog gameResultDialog;

    private int         curScore;

    private boolean isGameStarted = false;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_game_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
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
                }
            }
        });
        // controller button
        btnController = findViewById(R.id.activity_game_main_img_controller);
        btnController.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isGameStarted){
                    stopGame();
                } else {
                    startGame();
                }
                isGameStarted = !isGameStarted;
            }
        });
        // exit button
        btnExit = findViewById(R.id.activity_game_main_exit);
        btnExit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if( !isGameStarted ) {
                    finish();
                }
            }
        });

        // texture
        textureView = findViewById(R.id.activity_game_main_textureview);


        // prepare to start
        initGame();
        connectServer();
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
        GAME_MAX_SCORE = settings.getInt(this.KEY_MAX_SCORE, 0);
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
        tableRow1 = findViewById(R.id.activity_game_main_table_row1);
        tableRow2 = findViewById(R.id.activity_game_main_table_row2);
        tableRow3 = findViewById(R.id.activity_game_main_table_row3);
        LinearLayout[] rows = new LinearLayout[] {tableRow1, tableRow2, tableRow3};
        LinearLayout.LayoutParams cellParam = new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.MATCH_PARENT,
                1/(float)colNum);
        RelativeLayout.LayoutParams cellChildrenParam = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.MATCH_PARENT,
                RelativeLayout.LayoutParams.MATCH_PARENT);
        cellChildrenParam.addRule(RelativeLayout.CENTER_IN_PARENT, RelativeLayout.TRUE);
        for(int rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
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
        takeImageHandler.post(takeImageRunnable);
    }

    private void stopGame(){
        btnController.setImageResource(R.drawable.button_main_start);
        if( GAME_MODE== VALUE_MODE_TIMER) {
            gameCounterDownTimer.cancel();
        }
        gameHandler.stopGame();
        showGameResult();
        takeImageHandler.removeCallbacks(takeImageRunnable);
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
    private GameServerConnector serverConnector;
    private Toast toast;

    private void connectServer(){
        SharedPreferences settings = getSharedPreferences(MainActivity.PREFERENCE_NAME, Context.MODE_PRIVATE);
        String lastIp = settings.getString(GameSettingActivity.BUNDLE_KEY_IP, null);
        String lastPort = settings.getString(GameSettingActivity.BUNDLE_KEY_PORT, "0");
        serverConnector = new GameServerConnector(lastIp, Integer.parseInt(lastPort));
        serverConnector.setConnectCallback(new GameServerConnector.StringCallback() {
            @Override
            public void onResponse(String str) {
                if( toast!=null ) {
                    toast.cancel();
                }
                imgServerConnect.setImageResource(R.drawable.game_main_server_valid);
                if( isGameStarted ){
                    takeImageHandler.post(takeImageRunnable);
                }
            }

            @Override
            public void onError(String str) {
                imgServerConnect.setImageResource(R.drawable.game_main_server_invalid);
                if (str.equalsIgnoreCase(GameServerConnector.ERROR_SETTING)) {
                    showToast("Please set the address and the port correctly");
                }
            }
        });
        serverConnector.socketCreate();
        serverConnector.setUiThreadHandler(new GameServerConnector.StringCallback() {
            @Override
            public void onResponse(String str) {
                imgServerConnect.setImageResource(R.drawable.game_main_server_valid);
                try {
                    if( str!=null ) {
                        JSONObject object = new JSONObject(str);
                        if (object != null ){
                            if (object.getBoolean("Valid")) {
                                analyzeGaze(object);
                                Log.d(LOG_TAG, object.toString());
                            } else {
                                showToast("No detection");
                                Log.d(LOG_TAG, "No detection");
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
                } else if (str.equalsIgnoreCase(GameServerConnector.ERROR_TIMEOUT)) {
                    imgServerConnect.setImageResource(R.drawable.game_main_server_timeout);
                }
            }
        });
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
    private Configuration   confHandler = Configuration.getInstance(this);
    private TextureView     textureView;
    private Handler         takeImageHandler = new Handler();
    private Runnable        takeImageRunnable;
    private Handler         uiThreadHandler = null;
    private int             prevReceivedGazeIndex = 0;
    private ArrayList<Point> estimationList = new ArrayList<>();


    private void cameraInit(){
        cameraHandler = CameraHandler.getInstance(this, true);
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
    private final String JSON_KEY_PREDICT_X = "PredictX";
    private final String JSON_KEY_PREDICT_Y = "PredictY";
    private final String JSON_KEY_SEQ_NUMBER = "SequenceNumber";
    private ArrayList<Pair<Float,Float>> lastThreePoints = new ArrayList<>();
    private final int   CLICK_THRESHOLD = 85;

    private void analyzeGaze(JSONObject object){
        try {
            int receivedIdx = object.getInt(JSON_KEY_SEQ_NUMBER);
            if( receivedIdx > prevReceivedGazeIndex ){
                prevReceivedGazeIndex = receivedIdx;
                double portraitHori = object.getDouble(JSON_KEY_PREDICT_Y);
                double portraitVert = object.getDouble(JSON_KEY_PREDICT_X);
                float[] loc = new float[2];
                loc[0] = (float)((portraitHori + confHandler.getCameraOffsetX())/confHandler.getScreenSizeX());
                loc[1] = (float)((portraitVert + confHandler.getCameraOffsetY())/confHandler.getScreenSizeY());
                lastThreePoints.add(new Pair(loc[0], loc[1]));
                if( lastThreePoints.size()<3 ){
                    return;
                } else if (lastThreePoints.size()>3) {
                    lastThreePoints.remove(0);
                }
                // analyze last three points distribution
                float aveX = (lastThreePoints.get(0).first
                        + lastThreePoints.get(1).first
                        + lastThreePoints.get(2).first)/3;
                float aveY = (lastThreePoints.get(0).second
                        + lastThreePoints.get(1).second
                        + lastThreePoints.get(2).second)/3;
                double diff = 0;
                for(Pair<Float, Float> eachPoint : lastThreePoints){
                    diff += Math.sqrt((double)(eachPoint.first - aveX)*(eachPoint.first - aveX)
                            + (eachPoint.second - aveY)*(eachPoint.second - aveY));
                }
                diff /= 3;
                Log.d(LOG_TAG, String.valueOf(diff));
                if(diff < CLICK_THRESHOLD){
                    // Obtain MotionEvent object
                    long downTime = SystemClock.uptimeMillis();
                    long eventTime = SystemClock.uptimeMillis() + 100;
                    int metaState = 0;
                    MotionEvent motionEvent = MotionEvent.obtain(
                            downTime,
                            eventTime,
                            MotionEvent.ACTION_UP,
                            aveX,
                            aveY,
                            metaState
                    );
                    // Dispatch touch event to view
                    table.dispatchTouchEvent(motionEvent);
                } else {
                    Log.d(LOG_TAG, "Moving");
                }
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
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
