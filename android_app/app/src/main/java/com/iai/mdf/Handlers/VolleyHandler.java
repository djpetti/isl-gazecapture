package com.iai.mdf.Handlers;

import android.content.Context;
import android.util.Log;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.NetworkResponse;
import com.android.volley.ParseError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.HttpHeaderParser;
import com.android.volley.toolbox.JsonRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Map;

/**
 * Created by ShenWang on 10/20/2017.
 */

public class VolleyHandler {


    private final String LOG_TAG = "VolleyHandler";
    private static VolleyHandler myInstance;
    private Context ctxt;
    private RequestQueue    mRequestQueue;

    private VolleyHandler(Context context){
        ctxt = context;
        mRequestQueue = Volley.newRequestQueue(ctxt.getApplicationContext());
    }

    public static synchronized VolleyHandler getInstance(Context context){
        if( myInstance==null ){
            myInstance = new VolleyHandler(context);
        }
        return myInstance;
    }

    public <T> void addToRequestQueue(Request<T> req) {
        mRequestQueue.add(req);
    }



    /************************* Custom Functions *************************/
    public interface ServerCallback{
        void onResponse(NetworkResponse response);
    }



    public void uploadFileToServer(String addr, byte[] fileBytes, String tag, Response.Listener<JSONObject> successListener){
        //String REQUEST_URL = "http://ec2-54-236-72-209.compute-1.amazonaws.com/api/v1.0/upload";
        Log.d(LOG_TAG, "New Request");
        VolleyMultipartRequest request = new VolleyMultipartRequest(
                Request.Method.POST,
                addr,
                fileBytes,
                tag,
                successListener,
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        NetworkResponse response = error.networkResponse;
                        if(response != null && response.data != null){
                            Log.d(LOG_TAG, String.valueOf(response.statusCode) + new String(response.data));
                        }
                    }
                });
        request.setShouldCache(false);
        mRequestQueue.add(request);
    }



    public class VolleyMultipartRequest extends JsonRequest<JSONObject> {
        private final String twoHyphens = "--";
        private final String lineEnd = "\r\n";
        private final String boundary = "----MobileGazeApp" + System.currentTimeMillis();

        private Response.Listener<JSONObject> mListener;
        private Response.ErrorListener mErrorListener;
        private byte[] imageBytes;
        private String key;


        /**
         * Constructor with option method and default header configuration.
         *
         * @param method        method for now accept POST and GET only
         * @param url           request destination
         * @param imageData     byte array of the image
         * @param tag           the key for the data on server
         * @param listener      on success event handler
         * @param errorListener on error event handler
         */
        public VolleyMultipartRequest(int method, String url, byte[] imageData,
                                      String tag,
                                      Response.Listener<JSONObject> listener,
                                      Response.ErrorListener errorListener) {
            super(method, url, null, listener, errorListener);
            this.imageBytes = imageData;
            this.key = tag;
            this.mListener = listener;
            this.mErrorListener = errorListener;
            setShouldCache(false);
            setRetryPolicy(new DefaultRetryPolicy(DefaultRetryPolicy.DEFAULT_TIMEOUT_MS*2, 0, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        }

        @Override
        public Map<String, String> getHeaders() throws AuthFailureError {
            return super.getHeaders();
        }

        @Override
        public String getBodyContentType() {
            return "multipart/form-data; boundary=" + boundary;
        }

        @Override
        public byte[] getBody() {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            StringBuffer sb = new StringBuffer();
            try {
                sb.append( twoHyphens + boundary + lineEnd );
                sb.append( "Content-Disposition: form-data; name=\"" + key + "\"; filename=\"test.jpg\"" + lineEnd );
                sb.append( "Content-Type: " + "image/jpeg" + lineEnd );
                sb.append( lineEnd );
                bos.write(sb.toString().getBytes());
                bos.write(imageBytes);
                bos.write(lineEnd.toString().getBytes());
                bos.write( (twoHyphens + boundary + twoHyphens +lineEnd).toString().getBytes() );
                return bos.toByteArray();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }

        @Override
        protected Response<JSONObject> parseNetworkResponse(NetworkResponse response) {
            try {
                String jsonString = new String(response.data, HttpHeaderParser.parseCharset(response.headers));
                return Response.success(
                        new JSONObject(jsonString),
                        HttpHeaderParser.parseCacheHeaders(response));
            } catch (Exception e) {
                return Response.error(new ParseError(e));
            }
        }

        @Override
        public void deliverError(VolleyError error) {
            mErrorListener.onErrorResponse(error);
        }
    }



}
