package com.iai.mdf.DependenceClasses;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.util.DisplayMetrics;
import android.util.Log;

import java.util.StringTokenizer;


/**
 * Created by mou on 3/12/18.
 */

public class DeviceConfiguration {

    public static final int COLLECTION_CAPTURE_DELAY_MIN = 350;
    public static final int DEMO_CAPTURE_DELAY_MIN = 1;

    private final String LOG_TAG = "DeviceConfiguration";
    private static final String PREFERENCE_NAME = "isl_mobile_eye_gaze";
    private static final String KEY_CAMERA_POS_X = "camera_pos_x";
    private static final String KEY_CAMERA_POS_Y = "camera_pos_y";
    private static final String KEY_DISPLAY_SHORT_CM = "display_size_short_cm";
    private static final String KEY_DISPLAY_LONG_CM = "display_size_long_cm";
    private static final String KEY_DISPLAY_SHORT_RESO = "display_reso_short";
    private static final String KEY_DISPLAY_LONG_RESO = "display_reso_long";
    private static final String KEY_CALIBRAIION_SPEED = "calibration_speed";
    private static final String KEY_CALIBRAIION_RESULT = "calibration_result";
    private static final String KEY_CAPTURE_SPEED_COLLECTION = "capture_speed_collection";
    private static final String KEY_CAPTURE_SPEED_REALTIME = "capture_speed_realtime";
    private static final String KEY_VIDEO_COLLECTION_FPS = "video_collection_fps";
    private static final String KEY_CAPTURE_ROTATION = "picture_rotation";
    private static final String KEY_DOT_CANDIDATE_ROW = "dot_candidate_row";
    private static final String KEY_DOT_CANDIDATE_COL = "dot_candidate_col";



    /**
     * positve X is from left to right along short edge;
     * positve Y is from left to right along long edge;
     */


    private static DeviceConfiguration myInstance;
    private Context ctxt;
    private float   cameraOffsetPWidth;
    private float   cameraOffsetPHeight;
    private float   screenSizePWidth;
    private float   screenSizePHeight;
    private int     screenResoPWidth;
    private int     screenResoPHeight;
    private int     calibrationSpeed;
    private float[] calibrationMatrix;
    private int     collectionCaptureDelayTime;
    private int     demoCaptureDelayTime;
    private int     videoCollectionFPS;
    private int     imageRotation;
    private int     dotCandidateRow;
    private int     dotCandidateCol;



    private DeviceConfiguration(Context context) {
        ctxt = context;
        calibrationMatrix = new float[6];
    }

    public static synchronized DeviceConfiguration getInstance(Context context){
        if( myInstance==null ){
            myInstance = new DeviceConfiguration(context);
        }
        return myInstance;
    }




    public void loadConfiguration(){
        SharedPreferences settings = ctxt.getSharedPreferences(PREFERENCE_NAME, Context.MODE_PRIVATE);
        cameraOffsetPWidth = settings.getFloat(this.KEY_CAMERA_POS_X, 0);
        cameraOffsetPHeight = settings.getFloat(this.KEY_CAMERA_POS_Y, 0);
        screenSizePWidth = settings.getFloat(this.KEY_DISPLAY_SHORT_CM, 0);
        screenSizePHeight = settings.getFloat(this.KEY_DISPLAY_LONG_CM, 0);
        DisplayMetrics displayMetrics = new DisplayMetrics();
        ((Activity)ctxt).getWindowManager().getDefaultDisplay().getRealMetrics(displayMetrics);
        if (ctxt.getResources().getConfiguration().orientation == Configuration.ORIENTATION_PORTRAIT){
            screenResoPWidth = displayMetrics.widthPixels;
            screenResoPHeight = displayMetrics.heightPixels;
        } else {
            screenResoPWidth = displayMetrics.heightPixels;
            screenResoPHeight = displayMetrics.widthPixels;
        }
        calibrationSpeed = settings.getInt(this.KEY_CALIBRAIION_SPEED, 500);
        // 3x2 matrix
        StringTokenizer st = new StringTokenizer(settings.getString(this.KEY_CALIBRAIION_RESULT, "1,0,0,1,0,0"), ",");
        for (int i = 0; i < 6; i++) {
            calibrationMatrix[i] = Float.parseFloat(st.nextToken());
            Log.d(LOG_TAG, String.valueOf(calibrationMatrix[i]));
        }
        collectionCaptureDelayTime = settings.getInt(this.KEY_CAPTURE_SPEED_COLLECTION, 700);
        demoCaptureDelayTime = settings.getInt(this.KEY_CAPTURE_SPEED_REALTIME, 300);
        videoCollectionFPS = settings.getInt(this.KEY_VIDEO_COLLECTION_FPS, 30);
        imageRotation = settings.getInt(this.KEY_CAPTURE_ROTATION, 0);
        dotCandidateRow = settings.getInt(this.KEY_DOT_CANDIDATE_ROW, 5);
        dotCandidateCol = settings.getInt(this.KEY_DOT_CANDIDATE_COL, 6);
    }

    public void saveConfiguration(){
        SharedPreferences settings = ctxt.getSharedPreferences(PREFERENCE_NAME, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = settings.edit();
        editor.putFloat(this.KEY_CAMERA_POS_X, cameraOffsetPWidth);
        editor.putFloat(this.KEY_CAMERA_POS_Y, cameraOffsetPHeight);
        editor.putFloat(this.KEY_DISPLAY_SHORT_CM, screenSizePWidth);
        editor.putFloat(this.KEY_DISPLAY_LONG_CM, screenSizePHeight);
        editor.putInt(this.KEY_CALIBRAIION_SPEED, calibrationSpeed);
        StringBuilder caliMatStr = new StringBuilder();
        caliMatStr.append(calibrationMatrix[0]).append(",").append(calibrationMatrix[1]).append(",")
                .append(calibrationMatrix[2]).append(",").append(calibrationMatrix[3]).append(",")
                .append(calibrationMatrix[4]).append(",").append(calibrationMatrix[5]);
        editor.putString(this.KEY_CALIBRAIION_RESULT, caliMatStr.toString());
        editor.putInt(this.KEY_CAPTURE_SPEED_COLLECTION, collectionCaptureDelayTime);
        editor.putInt(this.KEY_CAPTURE_SPEED_REALTIME, demoCaptureDelayTime);
        editor.putInt(this.KEY_VIDEO_COLLECTION_FPS, videoCollectionFPS);
        editor.putInt(this.KEY_CAPTURE_ROTATION, imageRotation);
        editor.putInt(this.KEY_DOT_CANDIDATE_ROW, dotCandidateRow);
        editor.putInt(this.KEY_DOT_CANDIDATE_COL, dotCandidateCol);
        editor.commit();
    }





    public float getCameraOffsetPWidth() {
        return cameraOffsetPWidth;
    }

    public void setCameraOffsetPWidth(float cameraOffsetPWidth) {
        this.cameraOffsetPWidth = cameraOffsetPWidth;
    }

    public float getCameraOffsetPHeight() {
        return cameraOffsetPHeight;
    }

    public void setCameraOffsetPHeight(float cameraOffsetPHeight) {
        this.cameraOffsetPHeight = cameraOffsetPHeight;
    }

    public float getScreenSizePWidth() {
        return screenSizePWidth;
    }

    public void setScreenSizePWidth(float screenSizePWidth) {
        this.screenSizePWidth = screenSizePWidth;
    }

    public float getScreenSizePHeight() {
        return screenSizePHeight;
    }

    public void setScreenSizePHeight(float screenSizePHeight) {
        this.screenSizePHeight = screenSizePHeight;
    }

    public int getScreenResoPWidth() {
        return screenResoPWidth;
    }

    public void setScreenResoPWidth(int screenResoPWidth) {
        this.screenResoPWidth = screenResoPWidth;
    }

    public int getScreenResoPHeight() {
        return screenResoPHeight;
    }

    public void setScreenResoPHeight(int screenResoPHeight) {
        this.screenResoPHeight = screenResoPHeight;
    }

    public int getCalibrationSpeed() {
        return calibrationSpeed;
    }

    public void setCalibrationSpeed(int calibrationSpeed) {
        this.calibrationSpeed = calibrationSpeed;
    }

    public float[] getCalibrationMatrix() {
        return calibrationMatrix;
    }

    public void setCalibrationMatrix(float[] calibrationMatrix) {
        this.calibrationMatrix = calibrationMatrix;
    }

    public int getCollectionCaptureDelayTime() {
        return collectionCaptureDelayTime;
    }

    public void setCollectionCaptureDelayTime(int collectionCaptureDelayTime) {
        this.collectionCaptureDelayTime = collectionCaptureDelayTime;
    }

    public int getDemoCaptureDelayTime() {
        return demoCaptureDelayTime;
    }

    public void setDemoCaptureDelayTime(int demoCaptureDelayTime) {
        this.demoCaptureDelayTime = demoCaptureDelayTime;
    }

    public int getVideoCollectionFPS() {
        return videoCollectionFPS;
    }

    public void setVideoCollectionFPS(int videoCollectionFPS) {
        this.videoCollectionFPS = videoCollectionFPS;
    }

    public int getImageRotation() {
        return imageRotation;
    }

    public void setImageRotation(int imageRotation) {
        this.imageRotation = imageRotation;
    }

    public int getDotCandidateRow() {
        return dotCandidateRow;
    }

    public void setDotCandidateRow(int dotCandidateRow) {
        this.dotCandidateRow = dotCandidateRow;
    }

    public int getDotCandidateCol() {
        return dotCandidateCol;
    }

    public void setDotCandidateCol(int dotCandidateCol) {
        this.dotCandidateCol = dotCandidateCol;
    }
}
