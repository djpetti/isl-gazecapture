package com.iai.mdf;

/**
 * Created by pgao on 10/6/2017.
 */

public class FaceDetectionAPI {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }

    public FaceDetectionAPI() {
        initNative();
    }

    @Override
    protected void finalize() throws Throwable {
        deallocNative();
        super.finalize();
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native boolean    loadModel(String faceModelFile, String landmarkModelFile);
    public native int[]      detectFace(long matAddrGr, int minFaceSize, int maxFaceSize, boolean fastAlgorithm);
    public native double[]   detectLandmarks(long matAddrGr, int[] face);

    // Prepare float images for tensorflow processing
    public native float[]    prepareEyeImage(long matAddrGr, double[] landmarks, int side, long rects);
    public native float[]    prepareFaceImage(long matAddrGr, int[] face);
    
    // Various drawing functions
    public native void       draw(long matAddrRgba, int[] face, double[] landmarks);
    public native void       drawBoundingBox(long matAddrRgba, int[] bbx);
    public native void       drawLandmarks(long matAddrRgba, double[] landmarks);

    // Initialization and cleanup
    private native  int      initNative();
    private native  void     deallocNative();

    /*
     * We use a class initializer to allow the native code to cache some fields.
     */
    private long mNativeHandle;
}