package com.iai.mdf.Handlers;

import android.media.MediaScannerConnection;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

/**
 * Created by mou on 11/11/18.
 */

public class TextFileHanlder {
    private final static String LOG_TAG = "TextFileHandler";
    private final static String FOLDER_NAME = "Android_Gaze_Data";
    private final static String LOG_FILE_NAME = "logs.txt";

    public static boolean WriteLogIntoFile(String content){
        File rootFolder = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), FOLDER_NAME);
        if (!rootFolder.exists()){
            if (!rootFolder.mkdirs()){
                Log.d("App", "failed to root directory");
            }
        }
        File logFile = new File(rootFolder.getPath() + File.separator + LOG_FILE_NAME);
        try {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd  HH:mm:ss");
            String timeDate = sdf.format(new Date());
            FileOutputStream fOut = new FileOutputStream(logFile, true);
            OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut);
            myOutWriter.append("======= " + timeDate + " =======\n");
            myOutWriter.append(content);
            myOutWriter.append("\n\n");
            myOutWriter.close();
            fOut.flush();
            fOut.close();
        } catch (IOException e) {
            Log.e(LOG_TAG, "File write failed: " + e.toString());
        }
        return true;
    }


    public static boolean WriteLogIntoFile(ArrayList<float[]> content){
        File rootFolder = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), FOLDER_NAME);
        if (!rootFolder.exists()){
            if (!rootFolder.mkdirs()){
                Log.d("App", "failed to root directory");
            }
        }
        File logFile = new File(rootFolder.getPath() + File.separator + LOG_FILE_NAME);
        try {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd  HH:mm:ss");
            String timeDate = sdf.format(new Date());
            FileOutputStream fOut = new FileOutputStream(logFile, true);
            OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut);
            myOutWriter.append("======= " + timeDate + " =======\n");
            for ( int i=0; i < content.size(); i++){
                String xStr = String.format("%.04f", content.get(i)[0]);
                String yStr = String.format("%.04f", content.get(i)[1]);
                myOutWriter.append(xStr + " " + yStr + "\n");
            }
            myOutWriter.append("\n\n");
            myOutWriter.close();
            fOut.flush();
            fOut.close();
        } catch (IOException e) {
            Log.e(LOG_TAG, "File write failed: " + e.toString());
        }
        return true;
    }




}
