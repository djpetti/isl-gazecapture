package com.iai.mdf.Handlers;

import android.os.AsyncTask;
import android.util.Log;

import com.loopj.android.http.AsyncHttpClient;
import com.loopj.android.http.JsonHttpResponseHandler;
import com.loopj.android.http.RequestParams;
import com.squareup.okhttp.MediaType;
import com.squareup.okhttp.OkHttpClient;
import com.squareup.okhttp.Request;
import com.squareup.okhttp.RequestBody;
import com.squareup.okhttp.Response;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.InputStream;

import cz.msebera.android.httpclient.Header;

/**
 * Created by Mou on 10/21/2017.
 */

public class AmazonServerHandler {

    private final String LOG_TAG = "AmazonServerHandler";
    private static final String IMAGE_UPLOAD_URL = "http://ec2-54-236-72-209.compute-1.amazonaws.com/api/v1.0/upload";
    private static final String IMAGE_UPLOAD_KEY = "image";


    private static AmazonServerHandler myInstance;
    AsyncHttpClient client;
    OkHttpClient okHttpClient;


    private AmazonServerHandler(){
        client = new AsyncHttpClient();
    }

    public static synchronized AmazonServerHandler getInstance(){
        if( myInstance==null ){
            myInstance = new AmazonServerHandler();
        }
        return myInstance;
    }



    public void uploadImage(byte[] imageBytes){
        InputStream imageInputStream = new ByteArrayInputStream(imageBytes);
        RequestParams params = new RequestParams();
        params.put(IMAGE_UPLOAD_KEY, imageInputStream);
        Log.d(LOG_TAG, "ready to Post");
        client.post(IMAGE_UPLOAD_URL, params, new JsonHttpResponseHandler() {
            @Override
            public void onSuccess(int statusCode, Header[] headers, JSONObject response) {
                // If the response is JSONObject instead of expected JSONArray
                Log.d(LOG_TAG, "JsonObject Response");
            }

            @Override
            public void onSuccess(int statusCode, Header[] headers, JSONArray timeline) {
                Log.d(LOG_TAG, "JsonArray Response");
            }
        });
    }

    public void test2(final byte[] imageBytes){
        String REQUEST_URL = "http://ec2-54-236-72-209.compute-1.amazonaws.com/api/v1.0/upload";
        PostTask task = new PostTask();
        task.execute(imageBytes);
    }

    public class PostTask extends AsyncTask<byte[], Void, String> {

        protected String doInBackground(byte[]... images) {
            try {
                String bodyJson = "{\"image\": \"" + new String(images[0]) + "\"}";
                Log.d(LOG_TAG, bodyJson);
                RequestBody body = RequestBody.create(MediaType.parse("image/jpeg; charset=utf-8"), bodyJson);
                Request request = new Request.Builder()
                        .url("http://ec2-54-236-72-209.compute-1.amazonaws.com/api/v1.0/upload")
                        .post(body)
                        .build();
                Response response = okHttpClient.newCall(request).execute();
                return response.body().string();
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        protected void onPostExecute(String getResponse) {
            try {
                JSONObject jsonRes = new JSONObject(getResponse);
                String message = jsonRes.getString("msg");
                if( !message.equalsIgnoreCase("success") ){
                    Log.d(LOG_TAG, message);
                } else {
                    JSONObject jsonData = jsonRes.getJSONObject("data");
                    String str = jsonData.getString("width") + "   " + jsonData.getString("height");
                    Log.d(LOG_TAG, str);
                }
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }

    }






}
