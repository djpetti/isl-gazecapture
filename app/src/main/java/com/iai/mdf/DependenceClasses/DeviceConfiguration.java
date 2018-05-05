package com.iai.mdf.DependenceClasses;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.util.DisplayMetrics;



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
    private static final String KEY_CAPTURE_SPEED_COLLECTION = "capture_speed_collection";
    private static final String KEY_CAPTURE_SPEED_REALTIME = "capture_speed_realtime";
    private static final String KEY_CAPTURE_ROTATION = "picture_rotation";



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
    private int     collectionCaptureDelayTime;
    private int     demoCaptureDelayTime;
    private int     imageRotation;



    private DeviceConfiguration(Context context) {
        ctxt = context;
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
        collectionCaptureDelayTime = settings.getInt(this.KEY_CAPTURE_SPEED_COLLECTION, 700);
        demoCaptureDelayTime = settings.getInt(this.KEY_CAPTURE_SPEED_REALTIME, 300);
        imageRotation = settings.getInt(this.KEY_CAPTURE_ROTATION, 0);
    }

    public void saveConfiguration(){
        SharedPreferences settings = ctxt.getSharedPreferences(PREFERENCE_NAME, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = settings.edit();
        editor.putFloat(this.KEY_CAMERA_POS_X, cameraOffsetPWidth);
        editor.putFloat(this.KEY_CAMERA_POS_Y, cameraOffsetPHeight);
        editor.putFloat(this.KEY_DISPLAY_SHORT_CM, screenSizePWidth);
        editor.putFloat(this.KEY_DISPLAY_LONG_CM, screenSizePHeight);
        editor.putInt(this.KEY_CAPTURE_SPEED_COLLECTION, collectionCaptureDelayTime);
        editor.putInt(this.KEY_CAPTURE_SPEED_REALTIME, demoCaptureDelayTime);
        editor.putInt(this.KEY_CAPTURE_ROTATION, imageRotation);
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

    public int getImageRotation() {
        return imageRotation;
    }

    public void setImageRotation(int imageRotation) {
        this.imageRotation = imageRotation;
    }
}
