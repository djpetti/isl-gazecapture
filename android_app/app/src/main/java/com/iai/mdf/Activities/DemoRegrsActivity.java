package com.iai.mdf.Activities;

import android.content.pm.ActivityInfo;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
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

//import com.moutaigua.isl_android_gaze.FaceDetectionAPI;
import com.iai.mdf.FaceDetectionAPI;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.Handlers.TimerHandler;
import com.iai.mdf.Handlers.VolleyHandler;
import com.iai.mdf.Handlers.ImageProcessHandler;
import com.iai.mdf.Handlers.TensorFlowHandler;
import com.iai.mdf.R;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;

/**
 * Created by Mou on 9/22/2017.
 */

public class DemoRegrsActivity extends AppCompatActivity {

    private final String LOG_TAG = "DemoRegrsActivity";
    private CameraHandler cameraHandler;
    private DrawHandler drawHandler;
    private TextureView textureView;
    private Spinner     spinnerView;
    private FrameLayout frame_background_grid;
    private FrameLayout view_dot_container;
    private FrameLayout frame_gaze_result;
    private FrameLayout frame_bounding_box;
    private TextView    result_board;
    private int[]       SCREEN_SIZE;
    private int[]       TEXTURE_SIZE;
    private FaceDetectionAPI detectionAPI;
    private BaseLoaderCallback openCVLoaderCallback;
    private boolean isRealTimeDetection = false;
    private Handler autoDetectionHandler = new Handler();
    private Runnable autoDetectionRunnable;
    private int     captureInterval = 450;
    private double[]    theFaces = new double[4];
    private TensorFlowHandler tensorFlowHandler;
    private int         mFrameIndex = 0;
    private int         currentClassNum = 4;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_demo_1);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        getSupportActionBar().hide();

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


        view_dot_container = (FrameLayout) findViewById(R.id.activity_demo_layout_dotHolder_background);
        frame_background_grid = (FrameLayout) findViewById(R.id.activity_demo_layout_background_grid);
        frame_background_grid.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));
        frame_background_grid.bringToFront();
        frame_gaze_result = (FrameLayout) findViewById(R.id.activity_demo_layout_dotHolder_result);
        frame_gaze_result.bringToFront();
        drawHandler = new DrawHandler(this, fetchScreenSize());
        drawHandler.setDotHolderLayout(view_dot_container);
//        drawHandler.showAllCandidateDots();

        frame_bounding_box = (FrameLayout) findViewById(R.id.activity_demo_layout_bounding_box);
//        frame_bounding_box.bringToFront();
        frame_bounding_box.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                result_board.setText("");
                drawHandler.clear(frame_bounding_box);
                drawHandler.clear(frame_gaze_result);
                Log.d(LOG_TAG, "pressed");  //cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
                isRealTimeDetection = !isRealTimeDetection;
                if(isRealTimeDetection){
                    frame_background_grid.setBackgroundColor(0xFFFFFFFF);   // cover texture with white
                    autoDetectionHandler.post(autoDetectionRunnable);
                } else {
                    frame_background_grid.setBackgroundColor(0x00FFFFFF);   // uncover texture with translucent
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    autoDetectionHandler.removeCallbacks(autoDetectionRunnable);
                    result_board.setText("Detection is Off\nPress Anywhere to Start");
                    initFaceArray(theFaces);    // clear saved faces
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
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
                Log.d(LOG_TAG, "Selected Nothing ");
            }
        });
        spinnerView.bringToFront();


        result_board = (TextView) findViewById(R.id.activity_demo_txtview_result);
        result_board.setText("Press Anywhere to Start");

        autoDetectionRunnable = new Runnable() {
            @Override
            public void run() {
                cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
//                drawHandler.clear(frame_bounding_box);
//                drawHandler.clear(frame_gaze_result);
                autoDetectionHandler.postDelayed(this, captureInterval);
            }
        };
        initFaceArray(theFaces);
        tensorFlowHandler = new TensorFlowHandler(this);
        tensorFlowHandler.pickModel(TensorFlowHandler.MODEL_ISL_FILE_NAME);



        // load model
        detectionAPI = new FaceDetectionAPI();
        Log.i(LOG_TAG, "Loading face models ...");
        String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        if (!detectionAPI.loadModel(
                "/"+ base + "/Download/face_det_model_vtti.model",
                "/"+ base + "/Download/model_landmark_49_vtti.model"
        )) {
            Log.d(LOG_TAG, "Error reading model files.");
        }
        // for saving temporary images and data
//        File file0 = new File("/sdcard/Download/gazeTest/");
//        if( !file0.exists() ){
//            file0.mkdir();
//            MediaScannerConnection.scanFile(this, new String[] {file0.getAbsolutePath()}, null, null);
//        }
//        File file1 = new File("/sdcard/Download/gazeTest/faces.txt");
//        if( file1.exists() ) {
//            file1.delete();
//        }
//        File file2 = new File("/sdcard/Download/gazeTest/landmarks.txt");
//        if( file2.exists() ) {
//            file2.delete();
//        }
//        File file3 = new File("/sdcard/Download/gazeTest/estimations.txt");
//        if( file3.exists() ) {
//            file3.delete();
//        }
//        File file4 = new File("/sdcard/Download/gazeTest/tfInputs.txt");
//        if( file4.exists() ) {
//            file4.delete();
//        }


//        float[][] input = readTfInputArray("/sdcard/Download/rt_tfInput2.dat");
//        for(int i=0; ; ++i){
//            float[] prb = tensorFlowHandler.getClassificationResult(input[i], new int[]{36, 60, 3});
//            Log.d(LOG_TAG, String.valueOf(prb[0]) + " " + String.valueOf(prb[1]) + " " + String.valueOf(prb[2]) + " " + String.valueOf(prb[3]));
//        }

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
                    drawHandler.clear(frame_bounding_box);
                    drawHandler.clear(frame_gaze_result);
                    TimerHandler.getInstance().tic();
                    float[] res = getGazeEstimation(image);
                    Log.d("esti", "Estimation Time: " + String.valueOf(TimerHandler.getInstance().toc()));
                    drawResult(res);
                }
                image.close();
            }
        });
        cameraHandler.startPreview(textureView);
    }

    @Override
    public void onPause(){
        super.onPause();
        cameraHandler.stopPreview();
        autoDetectionHandler.removeCallbacks(autoDetectionRunnable);
        initFaceArray(theFaces);    // clear saved faces
        isRealTimeDetection = false;
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

    private int getQuatNum(double[] array){
        for (int i=0; i<array.length; i++) {
            if( array[i]<0 ){
                return i/4;
            }
        }
        return array.length/4;
    }

    private void initFaceArray(double[] faceArray){
        for (int i = 0; i < faceArray.length; i++) {
            faceArray[i] = -1;
        }
    }

    private boolean fullDetection(Image image, float[] eyeRegion){
        String msg;
        boolean isEyeDetected = false;
        double[] theEyes = new double[]{-1, -1, -1, -1, -1, -1, -1, -1};
        int numOfSavedFace = getQuatNum(theFaces);
        if( numOfSavedFace==0 ){
            ImageProcessHandler.doFaceEyeDetection(image, theFaces, theEyes, eyeRegion);
        } else {
            ImageProcessHandler.doFaceTracking(image, theFaces, theEyes, eyeRegion);
        }
        Log.d(LOG_TAG, String.valueOf(isEyeDetected));
        // display result
        int numOfFaces = getQuatNum(theFaces);
        if( numOfFaces == 0 ){
            Log.d(LOG_TAG, "Face Not Found");
            msg = "Face Not Found";
        } else {
            Log.d(LOG_TAG, "Face Detected");
            msg = "Face has been detected\n";
            drawHandler.clear(frame_bounding_box);
            int bounding_box_x = (int) (TEXTURE_SIZE[0] * theFaces[0]);
            int bounding_box_y = (int) (TEXTURE_SIZE[1] * theFaces[1]);
            int bounding_box_width = (int) (TEXTURE_SIZE[0] * theFaces[2]);
            int bounding_box_height = (int) (TEXTURE_SIZE[1] * theFaces[3]);
            drawHandler.showBoundingBox(
                    bounding_box_x,
                    bounding_box_y,
                    bounding_box_width,
                    bounding_box_height,
                    TEXTURE_SIZE[0],
                    frame_bounding_box,
                    true);
            int numOfEyes = getQuatNum(theEyes);
            Log.d(LOG_TAG, String.valueOf(numOfFaces) + " eye has been detected");
            msg = msg + String.valueOf(numOfEyes) + " eye has been detected";
            for(int i=0; i<numOfEyes; i++) {
                bounding_box_x = (int) (TEXTURE_SIZE[0] * theEyes[i*4]);
                bounding_box_y = (int) (TEXTURE_SIZE[1] * theEyes[i*4+1]);
                bounding_box_width = (int) (TEXTURE_SIZE[0] * theEyes[i*4+2]);
                bounding_box_height = (int) (TEXTURE_SIZE[1] * theEyes[i*4+3]);
                drawHandler.showBoundingBox(
                        bounding_box_x,
                        bounding_box_y,
                        bounding_box_width,
                        bounding_box_height,
                        TEXTURE_SIZE[0],
                        frame_bounding_box,
                        true);
            }
        }
//        initFaceArray(theFaces);
        result_board.setText(msg);
        return isEyeDetected;
    }

    private float[] getGazeEstimation(Image image){
//        TimerHandler.getInstance().tic();
        Mat colorImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC4);
        ImageProcessHandler.getRGBMat(image, colorImg.getNativeObjAddr());
        Imgproc.cvtColor(colorImg, colorImg, Imgproc.COLOR_BGRA2BGR);
//        Log.d(LOG_TAG, "Format Conversion: " + String.valueOf(TimerHandler.getInstance().toc()));
//        TimerHandler.getInstance().tic();
        Mat grayImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC1);
        Imgproc.cvtColor(colorImg, grayImg, Imgproc.COLOR_BGR2GRAY);
//        Log.d(LOG_TAG, "Color -> Gray: " + String.valueOf(TimerHandler.getInstance().toc()));
//        TimerHandler.getInstance().tic();
        int[] face = detectionAPI.detectFace(grayImg.getNativeObjAddr(), 30, 300, true);
//        Log.d(LOG_TAG, "Face Detection: " + String.valueOf(TimerHandler.getInstance().toc()));
        if( face!=null ){
            double[] faceRatio = new double[]{
                    (double)face[0]/grayImg.cols(),
                    (double)face[1]/grayImg.rows(),
                    (double)face[2]/grayImg.cols(),
                    (double)face[3]/grayImg.rows()
            };
            drawHandler.showBoundingBoxInLandscape(faceRatio, TEXTURE_SIZE, frame_bounding_box, true);
        }
        if( face!=null ){
//            TimerHandler.getInstance().tic();
            double[] landmarks = detectionAPI.detectLandmarks(grayImg.getNativeObjAddr(), face);
//            Log.d(LOG_TAG, "Landmark Detection: " + String.valueOf(TimerHandler.getInstance().toc()));
            if( landmarks!=null ){
                Mat cropMat = new Mat(36, 60, CvType.CV_8UC4);
                ArrayList<String> tfInputNodes = new ArrayList<>();
                ArrayList<float[]> tfInputs = new ArrayList<>();
                ArrayList<int[]> tfInputSizes = new ArrayList<>();
                float[] tfLEyeInputArray = new float[36*60*3];
                float[] tfREyeInputArray = new float[36*60*3];
//                TimerHandler.getInstance().tic();
                int[] lEyeRect = ImageProcessHandler.getEyeRegionCropRect(landmarks, grayImg.width(), grayImg.height(), true);
                int[] rEyeRect = ImageProcessHandler.getEyeRegionCropRect(landmarks, grayImg.width(), grayImg.height(), false);
                if (lEyeRect!=null && rEyeRect!=null) {
                    ImageProcessHandler.cropSingleRegion(colorImg.getNativeObjAddr(), lEyeRect, new int[]{36,60,3}, tfLEyeInputArray, cropMat.getNativeObjAddr());
                    tfInputNodes.add("leftEye");
                    tfInputs.add(tfLEyeInputArray);
                    tfInputSizes.add(new int[]{36, 60, 3});
                    ImageProcessHandler.cropSingleRegion(colorImg.getNativeObjAddr(), rEyeRect, new int[]{36,60,3}, tfREyeInputArray, cropMat.getNativeObjAddr());
                    tfInputNodes.add("rightEye");
                    tfInputs.add(tfREyeInputArray);
                    tfInputSizes.add(new int[]{36, 60, 3});
                    float[] faceGrid = ImageProcessHandler.faceRectToGridArray(face, new int[]{24, 32});
                    faceGrid = ImageProcessHandler.standardizeGridArray(faceGrid, new int[]{24, 32});
                    tfInputNodes.add("grid");
                    tfInputs.add(faceGrid);
                    tfInputSizes.add(new int[]{24, 32});
//                    Log.d(LOG_TAG, "Eye -> TensorFlowInput: " + String.valueOf(TimerHandler.getInstance().toc()));
//                    TimerHandler.getInstance().tic();
                    float[] loc = tensorFlowHandler.getEstimatedLocation(tfInputNodes, tfInputs, tfInputSizes);
//                    Log.d(LOG_TAG, "TensorFlow Inference: " + String.valueOf(TimerHandler.getInstance().toc()));
                    return loc;
                }
            }
        }
        return null;
    }


    private void drawResult(float[] estimateGaze){
        if( estimateGaze!=null ){
            switch (currentClassNum){
                case 4:
                    drawHandler.draw4ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result, false); break;
                case 6:
                    drawHandler.draw6ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result, false); break;
                case 9:
                    drawHandler.draw9ClassRegrsResult(estimateGaze, TEXTURE_SIZE, frame_gaze_result,  false); break;
            }
        }
    }









    private int[][][][] readUInt16DataFromMatFile(String filename, int pic_num){
        int[][][][] image = new int[pic_num][36][60][3];
        try{
            AssetManager assetManager = getResources().getAssets();
            InputStream inputStream = assetManager.open(filename);
            byte data[] = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            for(int colorLayer=0; colorLayer<3; ++colorLayer) {
                for (int col = 0; col < 60; ++col) {
                    for (int row = 0; row < 36; ++row) {
                        for(int pic = 0; pic < pic_num; ++pic) {
                            image[pic][row][col][colorLayer] = (int) data[pic + pic_num * row + pic_num * 36 * col + pic_num * 36 * 60 * colorLayer];
                        }
                    }
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        return image;
    }

    private int[][][] readEyeStateFromMatFile(String filename, int pic_num){
        int[][][] image = new int[pic_num][32][32];
        try{
            AssetManager assetManager = getResources().getAssets();
            InputStream inputStream = assetManager.open(filename);
            byte data[] = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            for (int col = 0; col < 32; ++col) {
                for (int row = 0; row < 32; ++row) {
                    for(int pic = 0; pic < pic_num; ++pic) {
                        image[pic][row][col] = data[pic + pic_num * row + pic_num * 32 * col ] & 0xFF;
                    }
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        return image;
    }

    private int[][] readResultFromMatFile(String fileName){
        int[][] image = new int[425][2];
        try{
            AssetManager assetManager = getResources().getAssets();
            InputStream inputStream = assetManager.open(fileName);
            byte data[] = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            for(int col=0; col<2; ++col) {
                for (int row = 0; row < 425; ++row) {
                    image[row][col] = ((data[row*2 + 1 + 850*col]<<8)& 0xFFFF) | data[row*2 + 850*col] & 0x00FF;
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        return image;
    }

    public float[][] readTfInputArray(String filePath) {
        String encoding = "UTF-8";
        File file = new File(filePath);
        BufferedReader bufferedReader = null;
        ArrayList<Float[]> inputArray = new ArrayList<>();
        try {
            InputStreamReader read = new InputStreamReader(new FileInputStream(file), encoding);
            bufferedReader = new BufferedReader(read);
            String str = null;
            int count =0;
            while ((str = bufferedReader.readLine()) != null) {
                count ++;
                Float[] eachInput = new Float[6480];
                String[] s = str.split(" ");
                for (int i = 0; i < s.length; i++) {
                    eachInput[i] = Float.valueOf(s[i]);
                }
                inputArray.add(eachInput);
                if( count>10) {
                    break;
                }
            }
            bufferedReader.close();
            read.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        float[][] content = new float[inputArray.size()][6480];
        for(int i=0; i<inputArray.size(); ++i){
            for(int j=0; j<6480; ++j){
                content[i][j] = inputArray.get(i)[j];
            }
        }
        return content;
    }

    private int[][][] convertMatToIntArray(Mat mat){
        int[][][] array = new int[mat.rows()][mat.cols()][3];
        return array;
    }

    private float[] convertPythonArrayToAndroidArray(float[] arr){
        float[] newArray = new float[36*60*3];
        for(int i=0; i<newArray.length; i++){
            int rgb = i / 36 / 60;
            int cr = i - rgb * 36 * 60;
            int c = cr / 36;
            int r = cr - c * 36;
            newArray[r * 60 * 3 + c * 3 + rgb] = arr[i];
        }
        return newArray;
    }

    private int[][][] readImageFromFile(String filename){
        String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        String imagePath = "/"+ base + "/Download/" + filename;
        File picFile = new File( imagePath );
        int size = (int)picFile.length();
        byte[] data = new byte[size];
        try {
            BufferedInputStream buf = new BufferedInputStream(new FileInputStream(picFile));
            buf.read(data, 0, data.length);
            buf.close();
        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
        int[][][] image = new int[36][60][3];
        for(int colorLayer=0; colorLayer<3; ++colorLayer) {
            for (int col = 0; col < 60; ++col) {
                for (int row = 0; row < 36; ++row) {
                        image[row][col][colorLayer] = (int) data[row + 4 * col + 36 * 60 * colorLayer];
                }
            }
        }
        return image;
    }

    private int[]   readImageFromFile2(String filename){
        String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        String imagePath = "/"+ base + "/Download/" + filename;
        Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
        int x = bitmap.getWidth();
        int y = bitmap.getHeight();
        int[]   intArray = new int[x*y];
//        bitmap.getPixel
        return  null;
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

    private void uploadImage(byte[] imageBytes){
        VolleyHandler.getInstance(DemoRegrsActivity.this).uploadFileToServer(
                "http://ec2-54-236-72-209.compute-1.amazonaws.com/api/v1.0/upload",
                imageBytes,
                "image",
                new com.android.volley.Response.Listener<JSONObject>(){
                    @Override
                    public void onResponse(JSONObject response) {
//                        android.os.Debug.waitForDebugger();
                        Log.d(LOG_TAG, String.valueOf(TimerHandler.getInstance().toc()) );
                        try {
                            JSONObject jsonRes = response;
                            String message = jsonRes.getString("msg");
                            if( !message.equalsIgnoreCase("success") ){
                                Log.d("VolleyHandler", message);
                                result_board.setText(message);
                            } else {
                                if( !jsonRes.isNull("data")) {
                                    JSONObject jsonData = jsonRes.getJSONObject("data");
                                    double widthRatio = jsonData.getDouble("width");
                                    double heightRatio = jsonData.getDouble("height");
                                    int x = (int) (SCREEN_SIZE[0] * widthRatio);
                                    int y = (int) (SCREEN_SIZE[1] * heightRatio);
                                    drawHandler.showDot(x, y, frame_gaze_result);
                                }
                                String timer1 = jsonRes.getString("timer1").substring(0,5);
                                String timer2 = jsonRes.getString("timer2").substring(0,5);
                                String resStr = "Prep: " + timer1 + "sec\n Crop: " + timer2 + " sec";
                                Log.d(LOG_TAG, resStr);
                                result_board.setText(resStr);
                            }
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                }
        );
    }






    /***************** Test *******************/

    public void saveImageByteIntoFile(byte[] imageData, String file_name){
        if(file_name==null || file_name.isEmpty()){
            Log.d(LOG_TAG, "Invalid filename. Image is not saved");
            return;
        }
        String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        String imagePath = "/"+ base + "/Download/" + file_name;
        File picFile = new File( imagePath );
        try {
            OutputStream output = new FileOutputStream(picFile);
            output.write(imageData);
            output.close();
        } catch (IOException e) {
            Log.e(LOG_TAG, "Exception occurred while saving picture to external storage ", e);
        }
    }

    public void saveBitmapIntoFile(Bitmap bitmap, String filename){
        FileOutputStream outFile = null;
        try {
            String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
            String imagePath = "/"+ base + "/Download/" + filename;
            outFile = new FileOutputStream(imagePath);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outFile); // bmp is your Bitmap instance
            // PNG is a lossless format, the compression factor (100) is ignored
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (outFile != null) {
                    outFile.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public byte[] readJPGIntoByteArray(String path){
        String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        String imagePath = "/"+ base + path;
        File file = new File(imagePath);
        int size = (int) file.length();
        byte[] bytes = new byte[size];
        try {
            BufferedInputStream buf = new BufferedInputStream(new FileInputStream(file));
            buf.read(bytes, 0, bytes.length);
            buf.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            return bytes;
        }
    }

    private void saveFace(int[] face){
        if( face!=null ) {
            File album = new File("/sdcard/Download/gazeTest/");
            if( album.isDirectory() || album.mkdir() ){
                String faceString = String.valueOf(mFrameIndex)+ " "
                        + String.valueOf(face[0])+ " "
                        + String.valueOf(face[1])+ " "
                        + String.valueOf(face[2])+ " "
                        + String.valueOf(face[3]) + "\n";
                try {
                    FileWriter fileWriter = new FileWriter("/sdcard/Download/gazeTest/faces.txt", true);
                    fileWriter.append(faceString);
                    fileWriter.close();
                }
                catch (IOException e) {
                    Log.e("Exception", "File write failed: " + e.toString());
                }
            } else {
                Log.d(LOG_TAG, "Directory doesn't exist");
            }
        }
    }

    private void saveLandmark(double[] landmark){
        if( landmark!=null ) {
            String landmarkString = String.valueOf(mFrameIndex);
            File album = new File("/sdcard/Download/gazeTest/");
            if( album.isDirectory() || album.mkdir() ){
                for(int i=0; i < landmark.length/2; i++) {
                    landmarkString += " " + String.valueOf(landmark[2*i]) + " " + String.valueOf(landmark[2*i+1]);
                }
                landmarkString += "\n";
                try {
                    FileWriter fileWriter = new FileWriter("/sdcard/Download/gazeTest/landmarks.txt", true);
                    fileWriter.append(landmarkString);
                    fileWriter.close();
                }
                catch (IOException e) {
                    Log.e("Exception", "File write failed: " + e.toString());
                }
            } else {
                Log.d(LOG_TAG, "Directory doesn't exist");
            }
        }
    }

    private void saveEstimation(float[] pointRatio){
        if( pointRatio!=null ) {
            String estiString = String.valueOf(mFrameIndex);
            File album = new File("/sdcard/Download/gazeTest/");
            if( album.isDirectory() || album.mkdir() ){
                estiString += " " + String.valueOf(pointRatio[0]) + " " + String.valueOf(pointRatio[1]) + "\n";
                try {
                    FileWriter fileWriter = new FileWriter("/sdcard/Download/gazeTest/estimations.txt", true);
                    fileWriter.append(estiString);
                    fileWriter.close();
                }
                catch (IOException e) {
                    Log.e("Exception", "File write failed: " + e.toString());
                }
            } else {
                Log.d(LOG_TAG, "Directory doesn't exist");
            }
        }
    }

    private void saveTensorFlowInputs(float[] inputs){
        if( inputs!=null ) {
            String inputString = String.valueOf(mFrameIndex);
            File album = new File("/sdcard/Download/gazeTest/");
            if( album.isDirectory() || album.mkdir() ){
                for(int i=0; i < inputs.length; i++) {
                    inputString += " " + String.valueOf(inputs[i]);
                }
                inputString += "\n";
                try {
                    FileWriter fileWriter = new FileWriter("/sdcard/Download/gazeTest/tfInputs.txt", true);
                    fileWriter.append(inputString);
                    fileWriter.close();
                }
                catch (IOException e) {
                    Log.e("Exception", "File write failed: " + e.toString());
                }
            } else {
                Log.d(LOG_TAG, "Directory doesn't exist");
            }
        }
    }

    private void saveFaceGrid(float[] inputs){
        if( inputs!=null ) {
            String inputString = String.valueOf(inputs[0]);
            for(int i=1; i < inputs.length; i++) {
                inputString += " " + String.valueOf(inputs[i]);
            }
            inputString += "\n";
            try {
                FileWriter fileWriter = new FileWriter("/sdcard/Download/gridNorm.dat", true);
                fileWriter.append(inputString);
                fileWriter.close();
            }
            catch (IOException e) {
                Log.e("Exception", "File write failed: " + e.toString());
            }
        }
    }

    private void saveDetection(int[] face, double[] landmarks, int index){
        try {
            FileWriter fileWriter = new FileWriter("/sdcard/Download/detection/"+String.valueOf(index)+".dat", false);
            fileWriter.append(
                    String.valueOf(face[0]) + " "
                            + String.valueOf(face[1]) + " "
                            + String.valueOf(face[2]) + " "
                            + String.valueOf(face[3]) + "\n"
            );
            for(int markIdx=0; markIdx<landmarks.length/2; markIdx++){
                fileWriter.append(String.valueOf(landmarks[markIdx*2]) + " " + String.valueOf(landmarks[markIdx*2+1]) + "\n");
            }
            fileWriter.close();
        } catch (IOException e) {
            Log.e("Exception", "File write failed: " + e.toString());
        }
    }

    private void saveTf4ClassResult(float[] loc) {
        try {
            FileWriter fileWriter = new FileWriter("/sdcard/Download/rt_tfOutput.dat", true);
            fileWriter.append(loc[0] + " " + loc[1] + " " + loc[2] + " " + loc[3] + "\n");
            fileWriter.close();
        } catch (IOException e) {
            Log.e("Exception", "File write failed: " + e.toString());
        }
    }



}
