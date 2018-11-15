package com.iai.mdf.Handlers;

import android.media.Image;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

import com.iai.mdf.Activities.DataCollectionActivity;
import com.iai.mdf.DependenceClasses.DeviceConfiguration;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
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
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Created by Mou on 2/23/2018.
 */

public class SocketHandler {

    public static final String SUCCESS_CONNECT_MSG = "connected";
    public static final String ERROR_DISCONNECTED = "disconnected";
    public static final String ERROR_SETTING = "setting";
    public static final String ERROR_TIMEOUT = "timeout";
    public static final String JSON_KEY_VALID = "Valid";
    public static final String JSON_KEY_PREDICT_X = "PredictX";
    public static final String JSON_KEY_PREDICT_Y = "PredictY";
    public static final String JSON_KEY_SEQ_NUMBER = "SequenceNumber";
    private static final String LOG_TAG = "SocketHandler";
    private static final int MSG_ON_SUCCESS = 1;
    private static final int MSG_ON_ERROR = 2;
    private static final int TIMEOUT_LENGTH = 1500;
    private static final int MAX_SUCCESSIVE_TIMEOUT = 80;


    private Socket mSocket = null;
    private OutputStream socketOutputStream;
    private DataOutputStream dos;
    private boolean isConnected = false;
    private Handler uiThreadHandler = null;
    private int    missingResponse;
    private String  serverAddr;
    private int     serverPort;
    private Semaphore writeLock;







    public interface StringCallback{
        void onResponse(String str);
        void onError(String str);
    }



    public SocketHandler(final String addr, final int port){
        this.serverAddr = addr;
        this.serverPort = port;
        writeLock = new Semaphore(1);
    }


    public void socketCreate(){
        missingResponse = 0;
        new Thread(){
            @Override
            public void run() {
                try {
                    mSocket = new Socket(serverAddr, serverPort);
                    mSocket.setSoTimeout(TIMEOUT_LENGTH);
                    socketOutputStream = mSocket.getOutputStream();
                    dos = new DataOutputStream(socketOutputStream);
                    isConnected = true;
                    Message errorMessage = uiThreadHandler.obtainMessage(MSG_ON_SUCCESS, SUCCESS_CONNECT_MSG);
                    errorMessage.sendToTarget();
                } catch (SocketTimeoutException e){
                    Log.d(LOG_TAG, "Create Socket Timeout");
                    isConnected = false;
                    Message errorMessage = uiThreadHandler.obtainMessage(MSG_ON_ERROR, ERROR_SETTING);
                    errorMessage.sendToTarget();
                } catch (IOException e) {
                    Log.d(LOG_TAG, "can't connect to the server");
                    isConnected = false;
                    Message errorMessage = uiThreadHandler.obtainMessage(MSG_ON_ERROR, ERROR_SETTING);
                    errorMessage.sendToTarget();
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
                mSocket = null;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void send(final byte[] imageBytes){
        if( mSocket!=null && mSocket.isConnected() ) {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        if( missingResponse < MAX_SUCCESSIVE_TIMEOUT) {
//                            semaphore.tryAcquire(2000, TimeUnit.MICROSECONDS);
                            missingResponse++;
                            writeLock.tryAcquire(TIMEOUT_LENGTH, TimeUnit.MICROSECONDS);
                            dos.write(imageBytes);
                            dos.flush();
                            writeLock.release();
                            Log.d(LOG_TAG, "Data Sent");
                            BufferedReader input = new BufferedReader(new InputStreamReader(mSocket.getInputStream()));
                            String res = input.readLine();
                            missingResponse = 0;
                            Log.d(LOG_TAG, "Missing Response: " + String.valueOf(missingResponse));
                            Message completeMessage = uiThreadHandler.obtainMessage(MSG_ON_SUCCESS, res);
                            completeMessage.sendToTarget();
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
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }).start();
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
                        Log.d(LOG_TAG, response);
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

    public void uploadImage(Image image, DeviceConfiguration confHandler){
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

    public void uploadImageOnBLU(Image image){
        TimerHandler.getInstance().tic();
        Mat colorImg = new Mat(
                DataCollectionActivity.Image_Size.getWidth(),
                DataCollectionActivity.Image_Size.getHeight(),
                CvType.CV_8UC4);
        ImageProcessHandler.getRGBMat(image, colorImg.getNativeObjAddr());
        Imgproc.cvtColor(colorImg, colorImg, Imgproc.COLOR_BGRA2BGR);
//
        byte[] jpegBytes = ImageProcessHandler.fromMatToJpegByte(colorImg);
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
