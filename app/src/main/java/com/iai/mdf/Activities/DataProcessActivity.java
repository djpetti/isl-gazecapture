package com.iai.mdf.Activities;

import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.iai.mdf.FaceDetectionAPI;
import com.iai.mdf.Handlers.ImageProcessHandler;
import com.iai.mdf.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by Mou on 11/25/2017.
 */

public class DataProcessActivity extends AppCompatActivity {


    private final String LOG_TAG = "DataProcessActivity";
    private final String ProcessedFolderName = "processedFolder";
    private final String DetectionResultFolderName = "detectionResult";
    private final String LeftEyeFolderName = "leftEye";
    private final String RightEyeFolderName = "rightEye";
    private final String CheckMissingFolderName = "Check";
    private final String NormalizedDataFileName = "normData.dat";
    private final String DataOrderFileName = "order.dat";
    private final String AbsoluteLocationFileName = "XY.dat";
    private final String CheckMissingFileName = "CheckMissing";
    private final int    ACTION_NONE = -1;
    private final int    ACTION_LEFT_EYE = 0;
    private final int    ACTION_RIGHT_EYE = 1;
    private final int    ACTION_CHECK_MISSING = 2;


    private ListView folderListView;
    private ListView actionListView;
    private TextView resultView;
    private ArrayList<String> dataFolders;
    private FolderArrayListAdapter folderAdapter;
    private String[] actionArray = {"Left Eye", "Right Eye"}; // if add one, also add ACTION_ , modify featureCrop(), isActionDone(), createActionFolder(),saveAbsLocation()
    private boolean isProcessingDone = true;
    private FaceDetectionAPI detectionAPI;
    private BaseLoaderCallback openCVLoaderCallback;
    private int selectedFolder   = -1;
    private int selectedAction   = ACTION_NONE;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_data_process);
        getSupportActionBar().hide();

        dataFolders = scanDataFolders();

        folderListView = (ListView) findViewById(R.id.activity_data_process_listview_subfolder);
        folderListView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
        folderAdapter = new FolderArrayListAdapter(this, dataFolders);
        folderListView.setAdapter(folderAdapter);
        folderListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                if( !isProcessingDone ){
                    Toast.makeText(getApplicationContext(), "Wait until the task is done", Toast.LENGTH_SHORT).show();
                    return;
                }
                selectedFolder = i;
                folderAdapter.notifyDataSetChanged();
            }
        });


        actionListView = (ListView) findViewById(R.id.activity_data_process_listview_action);
        ActionArrayListAdapter actionAdapter = new ActionArrayListAdapter(this, actionArray);
        actionListView.setAdapter(actionAdapter);
        actionListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                if( !isProcessingDone ){
                    Toast.makeText(getApplicationContext(), "Wait until the task is done", Toast.LENGTH_SHORT).show();
                    return;
                }
                if( selectedFolder==-1 ){
                    resultView.setText("Select a dataset");
                    return;
                }
                selectedAction = i;
                new ProcessTask().execute();
            }
        });

        resultView = (TextView) findViewById(R.id.activity_data_process_txt_result);


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
        // init openCV
        initOpenCV();


    }


    @Override
    public void onBackPressed() {
        if( !isProcessingDone ){
            Toast.makeText(this, "Wait until the task is done", Toast.LENGTH_SHORT).show();
        } else {
            super.onBackPressed();
        }
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

    private ArrayList<String> scanDataFolders(){
        ArrayList<String> subFolderList = new ArrayList<>();
        File rootFolder = new File("/sdcard/Pictures/Android_Gaze_Data");
        File[] files = rootFolder.listFiles();
        for (int i = 0; i < files.length; ++i) {
            File file = files[i];
            if (file.isDirectory()) {
                subFolderList.add(file.getAbsolutePath());
            }
        }
        Collections.sort(subFolderList);
        return subFolderList;
    }

    private void createProcessedFolder(String path){
        File processedFolder = new File(path + "/" + ProcessedFolderName);
        if( !processedFolder.exists() ){
            processedFolder.mkdir();
        }
    }

    private boolean isActionDoneBefore(String path, int action){
        boolean res = false;
        File file;
        switch (action){
            case ACTION_LEFT_EYE:
                file = new File(path + "/" + ProcessedFolderName + "/" + LeftEyeFolderName + "/" + NormalizedDataFileName);
                res = file.exists();
                break;
            case ACTION_RIGHT_EYE:
                file = new File(path + "/" + ProcessedFolderName + "/" + RightEyeFolderName + "/" + NormalizedDataFileName);
                res = file.exists();
                break;
            case ACTION_CHECK_MISSING:
                file = new File(path + "/" + ProcessedFolderName + "/" + CheckMissingFolderName + "/" + CheckMissingFileName);
                res = file.exists();
                break;
        }
        return res;
    }

    private void createActionFolder(String path, int action){
        File file;
        switch (action){
            case ACTION_LEFT_EYE:
                file = new File(path + "/" + ProcessedFolderName + "/" + LeftEyeFolderName);
                if( !file.exists() ){
                    file.mkdir();
                }
                break;
            case ACTION_RIGHT_EYE:
                file = new File(path + "/" + ProcessedFolderName + "/" + RightEyeFolderName);
                if( !file.exists() ){
                    file.mkdir();
                }
                break;
            case ACTION_CHECK_MISSING:
                file = new File(path + "/" + ProcessedFolderName + "/" + CheckMissingFolderName);
                if( !file.exists() ){
                    file.mkdir();
                }
                break;
        }
    }

    private boolean isDetectedBefore(String path){
        File detectionFolder = new File(path + "/" + ProcessedFolderName + "/" + DetectionResultFolderName);
        return detectionFolder.exists();
    }



    /**
     * Do face and landmark detection and sava them into a file
     * @param file The image file
     */
    private void doDetection(File file){
        // start detection
        Mat colorImg = Imgcodecs.imread(file.getAbsolutePath());
        Mat grayImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC1);
        Imgproc.cvtColor(colorImg, grayImg, Imgproc.COLOR_BGR2GRAY);
        int[] face = detectionAPI.detectFace(grayImg.getNativeObjAddr(), 30, 300, true);
        if (face != null) {
            double[] landmarks = detectionAPI.detectLandmarks(grayImg.getNativeObjAddr(), face);
            if (landmarks != null) {
                // to save the detection result
                String nameWithJPGExt = file.getName();
                String nameWithDatExt = nameWithJPGExt.substring(0, nameWithJPGExt.length()-4) + ".dat";
                try {
                    FileWriter fileWriter = new FileWriter(file.getParent() + '/' + ProcessedFolderName + "/" + DetectionResultFolderName + "/" + nameWithDatExt, false);
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
        }
    }

    private void featureCrop(File file, int featureIdx){
        // read detection results and mark the order
        BufferedReader reader = null;
        int[] face = new int[4];
        double[] landmarks = new double[98];
        try {
            String nameWithJPGExt = file.getName();
            String nameWithDatExt = nameWithJPGExt.substring(0, nameWithJPGExt.length()-4) + ".dat";
            File detectResultFile = new File(file.getParent() + '/' + ProcessedFolderName + "/" + DetectionResultFolderName + "/" + nameWithDatExt);
            FileInputStream fis = new FileInputStream(detectResultFile);
            reader = new BufferedReader(new InputStreamReader(fis));
            String[] bbString = reader.readLine().split(" ");
            for (int i = 0; i < face.length; i++) {
                face[i] = Integer.parseInt(bbString[i]);
            }
            for (int i = 0; i < landmarks.length/2; i++) {
                String[] ldmkString = reader.readLine().split(" ");
                landmarks[i*2] = Double.parseDouble(ldmkString[0]);
                landmarks[i*2+1] = Double.parseDouble(ldmkString[1]);
            }
        } catch (IOException e) {
            Log.w(LOG_TAG, "Face is not detected in " + file.getName());
            return;
        }
        // feature cropping
        Mat colorImg = Imgcodecs.imread(file.getAbsolutePath());
        String eyeImageName;
        int[] eyeRect;
        float[] tensorflowInput;
        Mat cropMat;
        switch (featureIdx) {
            case ACTION_LEFT_EYE:
                eyeRect = ImageProcessHandler.getEyeRegionCropRect(landmarks, colorImg.width(), colorImg.height(), true);
                tensorflowInput = new float[36*60*3];
                cropMat =  new Mat(36, 60, CvType.CV_8UC3);
                ImageProcessHandler.cropSingleRegionAndSaveTFInput(
                        colorImg.getNativeObjAddr(),
                        eyeRect, new int[]{36,60,3},
                        tensorflowInput,
                        cropMat.getNativeObjAddr(),
                        file.getParent() + '/' + ProcessedFolderName + "/" + LeftEyeFolderName + "/" + NormalizedDataFileName);
                eyeImageName = file.getParent() + "/"
                        + ProcessedFolderName + "/"
                        + LeftEyeFolderName + "/"
                        + file.getName();
                if( ! new File(eyeImageName).exists() ) {
                    Imgcodecs.imwrite(eyeImageName, cropMat);
                }
                saveTheOrder(file, selectedAction);
                saveAbsLocation(file, selectedAction);
                break;
            case ACTION_RIGHT_EYE:
                eyeRect = ImageProcessHandler.getEyeRegionCropRect(landmarks, colorImg.width(), colorImg.height(), false);
                tensorflowInput = new float[36*60*3];
                cropMat =  new Mat(36, 60, CvType.CV_8UC3);
                ImageProcessHandler.cropSingleRegionAndSaveTFInput(
                        colorImg.getNativeObjAddr(),
                        eyeRect, new int[]{36,60,3},
                        tensorflowInput,
                        cropMat.getNativeObjAddr(),
                        file.getParent() + '/' + ProcessedFolderName + "/" + RightEyeFolderName + "/" + NormalizedDataFileName);
                eyeImageName = file.getParent() + "/"
                        + ProcessedFolderName + "/"
                        + RightEyeFolderName + "/"
                        + file.getName();
                if( ! new File(eyeImageName).exists() ) {
                    Imgcodecs.imwrite(eyeImageName, cropMat);
                }
                saveTheOrder(file, selectedAction);
                saveAbsLocation(file, selectedAction);
                break;
            case ACTION_CHECK_MISSING:

                break;
        }
    }

    private void saveTheOrder(File file, int action) {
        if ( file != null ) {
            String nameWithExt = file.getName();
            nameWithExt = nameWithExt.substring(0, nameWithExt.length()-4) + ".dat";
            try {
                FileWriter fileWriter = null;
                switch (action){
                    case ACTION_LEFT_EYE:
                        fileWriter = new FileWriter(file.getParent() + '/' + ProcessedFolderName + "/" + LeftEyeFolderName + "/" + DataOrderFileName, true);
                        break;
                    case ACTION_RIGHT_EYE:
                        fileWriter = new FileWriter(file.getParent() + '/' + ProcessedFolderName + "/" + RightEyeFolderName + "/" + DataOrderFileName, true);
                        break;
                    case ACTION_CHECK_MISSING:
                        break;
                    default:
                        fileWriter = new FileWriter(file.getParent() + '/' + ProcessedFolderName + "/" + DataOrderFileName, true);
                }
                fileWriter.append(nameWithExt + "\n");
                fileWriter.close();
            } catch (IOException e) {
                Log.e("Exception", "File write failed: " + e.toString());
            }
        } else {
            Log.d(LOG_TAG, "Directory doesn't exist");
        }
    }

    private void saveAbsLocation(File file, int action) {
        if ( file != null ) {
            String nameWithExt = file.getName();
            String[] comp = nameWithExt.substring(0, nameWithExt.length()-4).split("_");
            try {
                FileWriter fileWriter = null;
                switch (action){
                    case ACTION_LEFT_EYE:
                        fileWriter = new FileWriter(file.getParent() + '/' + ProcessedFolderName + "/" + LeftEyeFolderName + "/" + AbsoluteLocationFileName, true);
                        break;
                    case ACTION_RIGHT_EYE:
                        fileWriter = new FileWriter(file.getParent() + '/' + ProcessedFolderName + "/" + RightEyeFolderName + "/" + AbsoluteLocationFileName, true);
                        break;
                    case ACTION_CHECK_MISSING:
                        break;
                    default:
                        fileWriter = new FileWriter(file.getParent() + '/' + ProcessedFolderName + "/" + AbsoluteLocationFileName, true);
                }
                fileWriter.append(comp[2] + " " + comp[3] + "\n");
                fileWriter.close();
            } catch (IOException e) {
                Log.e("Exception", "File write failed: " + e.toString());
            }
        } else {
            Log.d(LOG_TAG, "Directory doesn't exist");
        }
    }



    class FolderArrayListAdapter extends ArrayAdapter<String> {
        private Context ctxt;
        private ArrayList<String> data;

       public FolderArrayListAdapter(Context context, ArrayList<String> list){
            super(context, R.layout.listview_data_process_folderlist);
            this.data = list;
            this.ctxt = context;
       }

        @Override
        public int getCount() {
            return data.size();
        }

        @Nullable
        @Override
        public String getItem(int position) {
            return data.get(position);
        }

        @NonNull
        @Override
        public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
            View view;
            LayoutInflater inflater = LayoutInflater.from(this.ctxt);
            view = inflater.inflate(R.layout.listview_data_process_folderlist, null);    //set layout for displaying items
            TextView folderName = view.findViewById(R.id.listview_folderlist_name);
            String[] trees = data.get(position).split("/");
            folderName.setText(trees[trees.length-1]);
            if( selectedFolder==position ){
                CheckBox checkBox = view.findViewById(R.id.listview_folderlist_checkbox);
                checkBox.setChecked(true);
            }
            return view;
        }
    }

    class ActionArrayListAdapter extends ArrayAdapter<String> {
        private Context ctxt;
        private String[] data;

        public ActionArrayListAdapter(Context context, String[] array){
            super(context, R.layout.listview_data_process_actionlist);
            this.ctxt = context;
            this.data = array;
        }

        @Override
        public int getCount() {
            return data.length;
        }

        @Nullable
        @Override
        public String getItem(int position) {
            return data[position];
        }

        @NonNull
        @Override
        public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
            View view;
            LayoutInflater inflater = LayoutInflater.from(this.ctxt);
            view = inflater.inflate(R.layout.listview_data_process_actionlist, null);    //set layout for displaying items
            TextView folderName = view.findViewById(R.id.listview_actionlist_name);
            folderName.setText(data[position]);
            return view;
        }
    }


    class ProcessTask extends AsyncTask<Void, String, Void> {

        @Override
        protected Void doInBackground(Void... voids) {
            isProcessingDone = false;
            createProcessedFolder(dataFolders.get(selectedFolder));
            createActionFolder(dataFolders.get(selectedFolder), selectedAction);
//            if( isActionDoneBefore(dataFolders.get(selectedFolder), selectedAction) ){
//                publishProgress("Done Before");
//                return null;
//            } else {
//                createActionFolder(dataFolders.get(selectedFolder), selectedAction);
//            }
            boolean isDetected = isDetectedBefore(dataFolders.get(selectedFolder));
            if( !isDetected ){
                File detectionFolder = new File(dataFolders.get(selectedFolder) + "/" + ProcessedFolderName + "/" + DetectionResultFolderName);
                detectionFolder.mkdir();
            }
            File rawDataFolder = new File(dataFolders.get(selectedFolder));
            File[] files = rawDataFolder.listFiles();

            publishProgress("Done: 0/" +  String.valueOf(files.length - 1) );
            for (int i = 0; i < files.length; ++i) {
                File file = files[i];
                if (file.isDirectory()) {
                    continue;
                }
                if (!isDetected) {
                    doDetection(file);
                }
                featureCrop(file, selectedAction);
                publishProgress("Done: " + String.valueOf(i+1) + "/" +  String.valueOf(files.length - 1));
            }
            return null;
        }

        @Override
        protected void onProgressUpdate(String... strs) {
            resultView.setText(strs[0]);
            super.onProgressUpdate(strs);
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            isProcessingDone = true;
            selectedFolder = -1;
            selectedAction = ACTION_NONE;
            folderAdapter.notifyDataSetChanged();
        }
    }




}
