package com.iai.mdf.Handlers;

import android.icu.text.LocaleDisplayNames;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONStringer;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.net.SocketException;
import java.net.SocketTimeoutException;

/**
 * Created by Mou on 2/23/2018.
 */

public class SocketHandler {

    public static final String ERROR_CONNECTION_FAIL = "connection fail";
    public static final String ERROR_DISCONNECTED = "disconnected";
    private static final String LOG_TAG = "SocketHandler";
    private static final int MESSAGE_FROM_SERVER = 1;
    private static final int MESSAGE_OF_ERROR = 2;



    private Socket mSocket = null;
    private OutputStream socketOutputStream;
    private DataOutputStream dos;
    private Handler uiThreadHandler = null;
    private int    missingResponse = 0;





    public interface StringCallback{
        void onResponse(String str);
        void onError(String str);
    }



    public SocketHandler(final String addr, final int port){
        new Thread() {
            @Override
            public void run() {
                try {
                    mSocket = new Socket(addr, port);
                    mSocket.setSoTimeout(2000);
                    socketOutputStream = mSocket.getOutputStream();
                    dos = new DataOutputStream(socketOutputStream);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }.start();
    }


    public void close(){
        try {
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
                        if( missingResponse < 10) {
                            dos.write(imageBytes);
                            Log.d(LOG_TAG, "Data Sent");
                            missingResponse++;
                            BufferedReader input = new BufferedReader(new InputStreamReader(mSocket.getInputStream()));
                            String res = input.readLine();
                            Message completeMessage = uiThreadHandler.obtainMessage(MESSAGE_FROM_SERVER, res);
                            completeMessage.sendToTarget();
                            missingResponse--;
                            Log.d(LOG_TAG, "Missing Response: " + String.valueOf(missingResponse));
                        } else {
                            Message errorMessage = uiThreadHandler.obtainMessage(MESSAGE_OF_ERROR, ERROR_DISCONNECTED);
                            errorMessage.sendToTarget();
                        }
                    } catch (SocketException e){
                        if(e.getMessage().equalsIgnoreCase("Broken Pipe")){
                            Log.d(LOG_TAG,  e.getMessage());
                        } else if (e.getMessage().equalsIgnoreCase("Read Timed out")){
                            Log.d(LOG_TAG,  e.getMessage());
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }.start();
        }
    }

    public void listen( ) {
        if( uiThreadHandler==null ){
            Log.d(LOG_TAG, "set uiThreadHandler first. Call setUiThreadHandler()");
            return;
        }
        Handler firstHandler = new Handler();
        firstHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (mSocket==null || !mSocket.isConnected()){
                    Message errorMessage = uiThreadHandler.obtainMessage(MESSAGE_OF_ERROR, ERROR_CONNECTION_FAIL);
                    errorMessage.sendToTarget();
                }
            }
        }, 1000);
    }

    public void setUiThreadHandler(final StringCallback callback){
        uiThreadHandler = new Handler(Looper.getMainLooper()){
            @Override
            public void handleMessage(Message msg) {
                String response = (String)msg.obj;
                Log.d(LOG_TAG, msg.toString());
                switch (msg.what){
                    case MESSAGE_FROM_SERVER:
                        callback.onResponse(response);
                        break;
                    case MESSAGE_OF_ERROR:
                        Log.d(LOG_TAG, response);
                        callback.onError(response);
                        break;
                }
            }
        };
    }




}
