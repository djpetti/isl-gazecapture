package com.iai.mdf.Activities.Game;

import android.media.Image;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

import com.iai.mdf.Activities.DataCollectionActivity;
import com.iai.mdf.DependenceClasses.Configuration;
import com.iai.mdf.Handlers.ImageProcessHandler;
import com.iai.mdf.Handlers.TimerHandler;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by Mou on 2/23/2018.
 */

public class GameServerConnector {

    public static final String SUCCESS_CONNECTED = "connected";
    public static final String SUCCESS_DETECTED = "detected";
    public static final String ERROR_DISCONNECTED = "disconnected";
    public static final String ERROR_SETTING = "setting";
    public static final String ERROR_TIMEOUT = "timeout";
    public static final String ERROR_NO_DETECTION = "no_detection";
    private static final String LOG_TAG = "GameServerConnector";
    private static final int MSG_ON_SUCCESS = 1;
    private static final int MSG_ON_ERROR = 2;




    private Socket mSocket = null;
    private OutputStream socketOutputStream;
    private DataOutputStream dos;
    private boolean isConnected = false;
    private Handler uiThreadHandler = null;
    private int    missingResponse;
    private StringCallback connectCallback;
    private String  serverAddr;
    private int     serverPort;





    public interface StringCallback{
        void onResponse(String str);
        void onError(String str);
    }



    public GameServerConnector(String addr, int port){
        this.serverAddr = addr;
        this.serverPort = port;
    }

    public void setConnectCallback(StringCallback callback){
        connectCallback = callback;
    }

    public void socketCreate(){
        missingResponse = 0;
        if( serverAddr==null || serverPort==0 ){
            if (connectCallback!=null){
                connectCallback.onError(ERROR_SETTING);
            }
            return;
        }
        new Thread() {
            @Override
            public void run() {
                try {
                    mSocket = new Socket(serverAddr, serverPort);
                    mSocket.setSoTimeout(1500);
                    socketOutputStream = mSocket.getOutputStream();
                    dos = new DataOutputStream(socketOutputStream);
                    isConnected = true;
                    connectCallback.onResponse(SUCCESS_CONNECTED);
                } catch (SocketTimeoutException e){
                    Log.d(LOG_TAG, "Create Socket Timeout");
                    isConnected = false;
                    if (connectCallback!=null){
                        connectCallback.onError(ERROR_TIMEOUT);
                    } else {
                        e.printStackTrace();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }.start();
    }

    public void socketDestroy(){
        try {
            isConnected = false;
            if(dos != null) {
                dos.close();
            }
            if(socketOutputStream!=null) {
                socketOutputStream.close();
            }
            if(mSocket!=null) {
                mSocket.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }





    public void send(final byte[] imageBytes){
        if( mSocket!=null && mSocket.isConnected() ) {
            new Thread() {
                @Override
                public void run() {
                    try {
                        if( missingResponse < 20) {
                            missingResponse++;
                            dos.write(imageBytes);
                            Log.d(LOG_TAG, "Data Sent");
                            BufferedReader input = new BufferedReader(new InputStreamReader(mSocket.getInputStream()));
                            String res = input.readLine();
                            Message completeMessage = uiThreadHandler.obtainMessage(MSG_ON_SUCCESS, res);
                            completeMessage.sendToTarget();
                            missingResponse--;
                            Log.d(LOG_TAG, "Missing Responses: " + String.valueOf(missingResponse));
                        } else {
                            Message errorMessage = uiThreadHandler.obtainMessage(MSG_ON_ERROR, ERROR_DISCONNECTED);
                            errorMessage.sendToTarget();
                            isConnected = false;
                            socketDestroy();
                        }
                    } catch (SocketTimeoutException e) {
                        Log.d(LOG_TAG,  "Timeout Happened");
                        Message errorMessage = uiThreadHandler.obtainMessage(MSG_ON_ERROR, ERROR_TIMEOUT);
                        errorMessage.sendToTarget();
                    } catch (SocketException e){
                        Log.d(LOG_TAG,  "Broken Pipe");
                        Message errorMessage = uiThreadHandler.obtainMessage(MSG_ON_ERROR, ERROR_DISCONNECTED);
                        errorMessage.sendToTarget();
                        isConnected = false;
                        socketDestroy();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }.start();
        }
    }


    public void setUiThreadHandler(final StringCallback callback){
        uiThreadHandler = new Handler(Looper.getMainLooper()){
            @Override
            public void handleMessage(Message msg) {
                String response = (String)msg.obj;
                Log.d(LOG_TAG, msg.toString());
                switch (msg.what){
                    case MSG_ON_SUCCESS:
                        callback.onResponse(response);
                        break;
                    case MSG_ON_ERROR:
                        callback.onError(response);
                        break;
                }
            }
        };
    }


    public boolean isConnected() {
        return isConnected;
    }





    /******  Higher Level of API ******/
    private int             mFrameIndex = 0;

    public void uploadImage(Image image, Configuration   confHandler){
        Log.d(LOG_TAG, "Come on");
        Mat yuvMat = ImageProcessHandler.getBGRMatFromImage(image);
        Mat colorImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC3);
        Imgproc.cvtColor(yuvMat, colorImg, Imgproc.COLOR_YUV2BGR_I420);
//        switch (confHandler.getImageRotation()){
//            case 0:
//                break;
//            case 90:
//                Core.rotate(colorImg, colorImg, Core.ROTATE_90_CLOCKWISE);
//                break;
//            case 180:
//                Core.rotate(colorImg, colorImg, Core.ROTATE_180);
//                break;
//            case 270:
//                Core.rotate(colorImg, colorImg, Core.ROTATE_90_COUNTERCLOCKWISE);
//                break;
//            default:
//                break;
//        }
        TimerHandler.getInstance().tic();
        byte[] jpegBytes = ImageProcessHandler.fromMatToJpegByte(colorImg);
        Log.d(LOG_TAG, "Image Format Conversion: " + String.valueOf(TimerHandler.getInstance().toc()));
        byte[] sizeBytes = ByteBuffer.allocate(4).putInt(jpegBytes.length).order(ByteOrder.nativeOrder()).array();
        byte[] seqBytes = new byte[2];
        seqBytes[0] = (byte)(mFrameIndex & 0xFF);
        byte[] data = new byte[jpegBytes.length + 5];
        System.arraycopy(sizeBytes, 0, data, 0, 4);
        System.arraycopy(jpegBytes, 0, data, 4, jpegBytes.length);
        System.arraycopy(seqBytes, 0, data, 4 + jpegBytes.length, 1);
        send(data);
        mFrameIndex++;
    }


}
