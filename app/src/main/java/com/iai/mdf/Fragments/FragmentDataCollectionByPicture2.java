package com.iai.mdf.Fragments;

import android.content.pm.ActivityInfo;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.iai.mdf.Activities.DataCollectionActivity;
import com.iai.mdf.DependenceClasses.DeviceConfiguration;
import com.iai.mdf.Handlers.CameraHandler;
import com.iai.mdf.Handlers.DrawHandler;
import com.iai.mdf.Handlers.ImageProcessHandler;
import com.iai.mdf.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by mou on 9/16/17.
 */


public class FragmentDataCollectionByPicture2 extends Fragment {


    private final String LOG_TAG = "FragmentDataByPicture2";
    private CameraHandler cameraHandler;
    private TextureView textureView;
    private TextView    textView;
    private DrawHandler drawHandler;
    private FrameLayout dotHolderLayout;
    private int[]       SCREEN_SIZE;
    private int[]       TEXTURE_SIZE;
    private BaseLoaderCallback openCVLoaderCallback;
    private Handler     dotGeneratorHandler = new Handler();
    private Runnable    dotGeneratorRunnable;
    private int dotCounter = 0;
    private boolean isPicSaved = true;
    private boolean isStarted = false;
    private DeviceConfiguration confHandler = DeviceConfiguration.getInstance(getActivity());
    private String      FOLDER_NAME = null;
    private String      CUR_IMAGE_NAME = null;



    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        getActivity().setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        return inflater.inflate(R.layout.fragment_data_collection2, container, false);
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Texture
        SCREEN_SIZE = fetchScreenSize();
        textureView = getActivity().findViewById(R.id.fragment_data_collection2_preview_textureview);
        // ensure texture fill the screen with a certain ratio
        TEXTURE_SIZE = SCREEN_SIZE;
        int expected_height = TEXTURE_SIZE[0]*DataCollectionActivity.Image_Size.getHeight()/DataCollectionActivity.Image_Size.getWidth();
        if( expected_height< TEXTURE_SIZE[1] ){
            TEXTURE_SIZE[1] = expected_height;
        } else {
            TEXTURE_SIZE[0] = TEXTURE_SIZE[1]*DataCollectionActivity.Image_Size.getWidth()/DataCollectionActivity.Image_Size.getHeight();
        }
        textureView.setLayoutParams(new RelativeLayout.LayoutParams(TEXTURE_SIZE[0], TEXTURE_SIZE[1]));

        // dot generation
        dotHolderLayout = getActivity().findViewById(R.id.fragment_data_collection2_layout_dotHolder);
        dotHolderLayout.bringToFront();
        dotHolderLayout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawHandler.clear(dotHolderLayout);
                Log.d(LOG_TAG, "pressed");  //cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_STILL_CAPTURE);
                isStarted = !isStarted;
                if(isStarted){
                    dotHolderLayout.setBackgroundColor(0xFFFFFFFF);   // cover texture with white
                    dotGeneratorHandler.postDelayed(dotGeneratorRunnable, 500);
                } else {
                    dotHolderLayout.setBackgroundColor(0x00FFFFFF);   // uncover texture with translucent
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    dotGeneratorHandler.removeCallbacks(dotGeneratorRunnable);
                    dotCounter = Math.max(dotCounter, 3);
                    Toast.makeText(getActivity(), "Take " + String.valueOf(dotCounter - 3) + " pictures", Toast.LENGTH_LONG).show();
                    dotCounter = 0;
                    if( isPicSaved ) {
                        deleteCurImage();
                    }
                }
            }
        });
        drawHandler = new DrawHandler(getActivity(), fetchScreenSize());
        drawHandler.setDotHolderLayout(dotHolderLayout);
        dotGeneratorRunnable = new Runnable() {
            @Override
            public void run() {
                if( isPicSaved ) {
                    isPicSaved = false;
                    drawHandler.showNextPoint();
                    delayCapture(confHandler.getCollectionCaptureDelayTime());
                }
                dotGeneratorHandler.postDelayed(this,20);
            }
        };

        // textview
        textView = getActivity().findViewById(R.id.fragment_data_collection2_txtview_result);

        Toast.makeText(getActivity(), "First 3 dots don't count", Toast.LENGTH_LONG).show();

    }

    @Override
    public void onResume() {
        super.onResume();
        drawHandler = new DrawHandler(getActivity(), SCREEN_SIZE);
        drawHandler.setDotHolderLayout((dotHolderLayout));
        cameraHandler = new CameraHandler(getActivity(), true);
        cameraHandler.setOnImageAvailableListenerForPrev(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                Image image = reader.acquireNextImage();
                if( cameraHandler.getCameraState()==CameraHandler.CAMERA_STATE_STILL_CAPTURE ) {
                    cameraHandler.setCameraState(CameraHandler.CAMERA_STATE_PREVIEW);
                    Log.d(LOG_TAG, "Take a picture");
                    dotCounter++;
                    setCurrentImageName();
                    saveImage(image);
                    isPicSaved = true;
                    if( dotCounter < 4 ){
                        deleteCurImage();
                    }
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
        dotGeneratorHandler.removeCallbacks(dotGeneratorRunnable);
        isStarted = false;
    }



    private void setCurrentImageName(){
        if( FOLDER_NAME == null ) {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd");
            String timeDate = sdf.format(new Date());
            String subFolderName = timeDate;
            File picFolder = new File(Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_PICTURES), "Android_Gaze_Data" + File.separator + subFolderName);
            if (!picFolder.exists()) {
                if (!picFolder.mkdirs()) {
                    Log.d("App", "failed to socketCreate directory");
                }
                MediaScannerConnection.scanFile(getActivity(), new String[]{picFolder.getAbsolutePath()}, null, null);
            }
            FOLDER_NAME = picFolder.getAbsolutePath();
        }
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
        String timestamp = sdf.format(new Date());
        String picName = timestamp + "_" + drawHandler.getCurrDot().x + "_" + drawHandler.getCurrDot().y + ".jpg";
        Log.d(LOG_TAG, picName);
        CUR_IMAGE_NAME = FOLDER_NAME + File.separator + picName;
    }

    private void saveImage(Image image){
        Mat yuvMat = ImageProcessHandler.getBGRMatFromImage(image);
        Mat colorImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC3);
        Imgproc.cvtColor(yuvMat, colorImg, Imgproc.COLOR_YUV2BGR_I420);
        switch (confHandler.getImageRotation()){
            case 0:
                break;
            case 90:
//                Core.rotate(colorImg, colorImg, Core.ROTATE_90_CLOCKWISE);
                ImageProcessHandler.rotateImage(colorImg, Core.ROTATE_90_CLOCKWISE);
                break;
            case 180:
//                Core.rotate(colorImg, colorImg, Core.ROTATE_180);
                ImageProcessHandler.rotateImage(colorImg, Core.ROTATE_180);
                break;
            case 270:
//                Core.rotate(colorImg, colorImg, Core.ROTATE_90_COUNTERCLOCKWISE);
                ImageProcessHandler.rotateImage(colorImg, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            default:
                break;
        }
        Imgcodecs.imwrite(CUR_IMAGE_NAME, colorImg);
    }

    private void deleteCurImage(){
        File picFile = new File(CUR_IMAGE_NAME);
        if( picFile.exists() && picFile.delete() ){
            Log.d(LOG_TAG, CUR_IMAGE_NAME + " is deleted");
        }
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

    private int[] fetchScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getActivity().getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }

    public boolean isPicSaved(){
        return isPicSaved;
    }


}
