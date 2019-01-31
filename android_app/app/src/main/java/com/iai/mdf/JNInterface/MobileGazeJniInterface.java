package com.iai.mdf.JNInterface;

/**
 * Created by Mou on 11/3/2017.
 */

public class MobileGazeJniInterface {

    static {
        System.loadLibrary("MobileGazeNative");
//        System.loadLibrary("native-lib");
    }


    private final String LOG_TAG = "MobileGazeJniInterface";

    public MobileGazeJniInterface(){

    }


    public native void fromMatToByteArray(long imageAddr, byte[] byteArray);

    public native byte[] encodeIntoJpegArray(byte[] yuvBytes, int width, int height, byte[] encodedBytes);

    public native void getRGBMatImage(byte[] yuvBytes, int width, int height, long matAddrRgba);

    public native int[] getRotatedRGBImage(byte[] yBytes, byte[] uBytes, byte[] vBytes, int origWidth, int origHeight);

    public native void rotateImage(long addr, int rotate);

    public native void cropImage(long matAddr, int[] rect, int[] resize, float[] tensorFlowInput, long cropAddr); // resize[0] is width; resize[1] is height

    public native void cropImageAndSaveInput(long matAddr, int[] rect, int[] size, float[] tensorFlowInput, long cropAddr, String path);

    public native int[] faceDetection(byte[] yBytes, byte[] uBytes, byte[] vBytes, int origWidth, int origHeight);



    // return whether eye is detected
    public native void faceEyeDetection(byte[] yuvBytes, int width, int height, double[] faces, double[] eyes, float[] eyeRegion);

    // return whether eye is detected
    public native void faceTracking(byte[] yuvBytes, int width, int height, double[] faces, double[] eyes, float[] eyeRegion);



    public native void getAdditionRes(int a, int b, double[] faces, double[] eyes);

    public native String getWelcomeString();




}
