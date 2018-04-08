package com.iai.mdf.DependenceClasses;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.util.DisplayMetrics;



/**
 * Created by mou on 3/12/18.
 */

public class Configuration {

    public static final int COLLECTION_CAPTURE_DELAY_MIN = 350;
    public static final int DEMO_CAPTURE_DELAY_MIN = 250;

    private final String LOG_TAG = "Configuration";
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


    private static Configuration myInstance;
    private Context ctxt;
    private float   cameraOffsetX;
    private float   cameraOffsetY;
    private float   screenSizeX;
    private float   screenSizeY;
    private int     screenResoX;
    private int     screenResoY;
    private int     collectionCaptureDelayTime;
    private int     demoCaptureDelayTime;
    private int     imageRotation;



    private Configuration(Context context) {
        ctxt = context;
    }

    public static synchronized Configuration getInstance(Context context){
        if( myInstance==null ){
            myInstance = new Configuration(context);
        }
        return myInstance;
    }




    public void loadConfiguration(){
        SharedPreferences settings = ctxt.getSharedPreferences(PREFERENCE_NAME, Context.MODE_PRIVATE);
        cameraOffsetX = settings.getFloat(this.KEY_CAMERA_POS_X, 0);
        cameraOffsetY = settings.getFloat(this.KEY_CAMERA_POS_Y, 0);
        screenSizeX = settings.getFloat(this.KEY_DISPLAY_SHORT_CM, 0);
        screenSizeY = settings.getFloat(this.KEY_DISPLAY_LONG_CM, 0);
        DisplayMetrics displayMetrics = new DisplayMetrics();
        ((Activity)ctxt).getWindowManager().getDefaultDisplay().getRealMetrics(displayMetrics);
        screenResoX = displayMetrics.widthPixels;
        screenResoY = displayMetrics.heightPixels;
        collectionCaptureDelayTime = settings.getInt(this.KEY_CAPTURE_SPEED_COLLECTION, 700);
        demoCaptureDelayTime = settings.getInt(this.KEY_CAPTURE_SPEED_REALTIME, 300);
        imageRotation = settings.getInt(this.KEY_CAPTURE_ROTATION, 0);
    }

    public void saveConfiguration(){
        SharedPreferences settings = ctxt.getSharedPreferences(PREFERENCE_NAME, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = settings.edit();
        editor.putFloat(this.KEY_CAMERA_POS_X, cameraOffsetX);
        editor.putFloat(this.KEY_CAMERA_POS_Y, cameraOffsetY);
        editor.putFloat(this.KEY_DISPLAY_SHORT_CM, screenSizeX);
        editor.putFloat(this.KEY_DISPLAY_LONG_CM, screenSizeY);
        editor.putInt(this.KEY_CAPTURE_SPEED_COLLECTION, collectionCaptureDelayTime);
        editor.putInt(this.KEY_CAPTURE_SPEED_REALTIME, demoCaptureDelayTime);
        editor.putInt(this.KEY_CAPTURE_ROTATION, imageRotation);
        editor.commit();
    }





    public float getCameraOffsetX() {
        return cameraOffsetX;
    }

    public void setCameraOffsetX(float cameraOffsetX) {
        this.cameraOffsetX = cameraOffsetX;
    }

    public float getCameraOffsetY() {
        return cameraOffsetY;
    }

    public void setCameraOffsetY(float cameraOffsetY) {
        this.cameraOffsetY = cameraOffsetY;
    }

    public float getScreenSizeX() {
        return screenSizeX;
    }

    public void setScreenSizeX(float screenSizeX) {
        this.screenSizeX = screenSizeX;
    }

    public float getScreenSizeY() {
        return screenSizeY;
    }

    public void setScreenSizeY(float screenSizeY) {
        this.screenSizeY = screenSizeY;
    }

    public int getScreenResoX() {
        return screenResoX;
    }

    public void setScreenResoX(int screenResoX) {
        this.screenResoX = screenResoX;
    }

    public int getScreenResoY() {
        return screenResoY;
    }

    public void setScreenResoY(int screenResoY) {
        this.screenResoY = screenResoY;
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
