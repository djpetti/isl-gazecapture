package com.iai.mdf.Activities;

import android.content.Context;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.Toast;

import com.iai.mdf.DependenceClasses.DeviceConfiguration;
import com.iai.mdf.R;

/**
 * Created by mou on 3/12/18.
 */

public class SettingActivity extends AppCompatActivity {



    private static final String PREFERENCE_NAME = "isl_mobile_eye_gaze";
    private static final String LOG_TAG = "SettingActivity";

    private DeviceConfiguration confHandler;
    private EditText editCameraPosX;
    private EditText editCameraPosY;
    private EditText editDisplaySizeX;
    private EditText editDisplaySizeY;
    private EditText editCaptureSpeedCollection;
    private EditText editCaptureSpeedRealtime;
    private EditText editPictureRotation;

    private boolean editMode = false;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_setting);

        confHandler = DeviceConfiguration.getInstance(this);

        editCameraPosX = findViewById(R.id.setting_activity_editxt_camera_short);
        editCameraPosX.setText(String.valueOf(confHandler.getCameraOffsetPWidth()));
        editCameraPosX.clearFocus();
        editCameraPosY = findViewById(R.id.setting_activity_editxt_camera_long);
        editCameraPosY.setText(String.valueOf(confHandler.getCameraOffsetPHeight()));

        editDisplaySizeX = findViewById(R.id.setting_activity_editxt_display_short_cm);
        editDisplaySizeX.setText(String.valueOf(confHandler.getScreenSizePWidth()));
        editDisplaySizeY = findViewById(R.id.setting_activity_editxt_display_long_cm);
        editDisplaySizeY.setText(String.valueOf(confHandler.getScreenSizePHeight()));

        editCaptureSpeedCollection = findViewById(R.id.setting_activity_editxt_capture_speed_collection);
        editCaptureSpeedCollection.setText(String.valueOf(confHandler.getCollectionCaptureDelayTime()));
        editCaptureSpeedRealtime = findViewById(R.id.setting_activity_editxt_capture_speed_realtime);
        editCaptureSpeedRealtime.setText(String.valueOf(confHandler.getDemoCaptureDelayTime()));

        editPictureRotation = findViewById(R.id.setting_activity_editxt_image_rotation);
        editPictureRotation.setText(String.valueOf(confHandler.getImageRotation()));

        View view = this.getCurrentFocus();
        if (view != null) {
            InputMethodManager imm = (InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
            imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        if( !editMode ) {
            inflater.inflate(R.menu.menu_setting_viewing, menu);
        } else {
            inflater.inflate(R.menu.menu_setting_editing, menu);
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu_action_edit:
                setEditxtMode(true);
                editMode = !editMode;
                invalidateOptionsMenu();
                return true;

            case R.id.menu_action_save:
                if( isInputValid() ){
                    setEditxtMode(false);
                    updateSetting();
                    editMode = !editMode;
                    invalidateOptionsMenu();
                    Toast.makeText(this, "Saved", Toast.LENGTH_SHORT).show();
                }
                return true;

            default:
                return super.onOptionsItemSelected(item);

        }
    }


    @Override
    protected void onResume() {
        super.onResume();
        // Check if no view has focus:
        // if focused, hide input keyboard

    }

    @Override
    public void onBackPressed() {
        if( editMode ){
            Toast.makeText(this, "Save the changes first", Toast.LENGTH_SHORT).show();
        } else {
            super.onBackPressed();
        }
    }



    private boolean isInputValid(){
        if( editCameraPosX.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editCameraPosY.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editDisplaySizeX.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editDisplaySizeY.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editCaptureSpeedCollection.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( Integer.parseInt(editCaptureSpeedCollection.getText().toString()) < DeviceConfiguration.COLLECTION_CAPTURE_DELAY_MIN){
            Toast.makeText(this, "Collection Capture delay should be greater than 350 ms", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editCaptureSpeedRealtime.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( Integer.parseInt(editCaptureSpeedRealtime.getText().toString()) < DeviceConfiguration.DEMO_CAPTURE_DELAY_MIN){
            Toast.makeText(this, "Collection Capture delay should be greater than 250 ms", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editPictureRotation.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        int degree = Integer.parseInt(editPictureRotation.getText().toString());
        if(  degree!=0 && degree!=90 && degree!=180 && degree!=270 ){
            Toast.makeText(this, "Rotation should be one in {0, 90, 180, 270}", Toast.LENGTH_SHORT).show();
            return false;
        }
        return true;
    }


    private void setEditxtMode(boolean enabled){
        if( !enabled ) {
            editCameraPosX.setEnabled(enabled);
            editCameraPosY.setEnabled(enabled);
            editDisplaySizeX.setEnabled(enabled);
            editDisplaySizeY.setEnabled(enabled);
            editCaptureSpeedCollection.setEnabled(enabled);
            editCaptureSpeedRealtime.setEnabled(enabled);
            editPictureRotation.setEnabled(enabled);
            editCameraPosX.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCameraPosY.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editDisplaySizeX.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editDisplaySizeY.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCaptureSpeedCollection.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCaptureSpeedRealtime.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editPictureRotation.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCameraPosX.setFocusableInTouchMode(enabled);
            editCameraPosY.setFocusableInTouchMode(enabled);
            editDisplaySizeX.setFocusableInTouchMode(enabled);
            editDisplaySizeY.setFocusableInTouchMode(enabled);
            editCaptureSpeedCollection.setFocusableInTouchMode(enabled);
            editCaptureSpeedRealtime.setFocusableInTouchMode(enabled);
            editPictureRotation.setFocusableInTouchMode(enabled);
        } else {
            editCameraPosX.setEnabled(enabled);
            editCameraPosY.setEnabled(enabled);
            editDisplaySizeX.setEnabled(enabled);
            editDisplaySizeY.setEnabled(enabled);
            editCaptureSpeedCollection.setEnabled(enabled);
            editCaptureSpeedRealtime.setEnabled(enabled);
            editPictureRotation.setEnabled(enabled);
            editCameraPosX.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCameraPosY.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editDisplaySizeX.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editDisplaySizeY.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCaptureSpeedCollection.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCaptureSpeedRealtime.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editPictureRotation.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCameraPosX.setFocusableInTouchMode(enabled);
            editCameraPosY.setFocusableInTouchMode(enabled);
            editDisplaySizeX.setFocusableInTouchMode(enabled);
            editDisplaySizeY.setFocusableInTouchMode(enabled);
            editCaptureSpeedCollection.setFocusableInTouchMode(enabled);
            editCaptureSpeedRealtime.setFocusableInTouchMode(enabled);
            editPictureRotation.setFocusableInTouchMode(enabled);
        }
    }


    private void updateSetting(){
        confHandler.setCameraOffsetPWidth( Float.parseFloat(editCameraPosX.getText().toString()) ) ;
        confHandler.setCameraOffsetPHeight( Float.parseFloat(editCameraPosY.getText().toString()) ) ;
        confHandler.setScreenSizePWidth( Float.parseFloat(editDisplaySizeX.getText().toString()) ) ;
        confHandler.setScreenSizePHeight( Float.parseFloat(editDisplaySizeY.getText().toString()) ) ;
        confHandler.setCollectionCaptureDelayTime( Integer.parseInt(editCaptureSpeedCollection.getText().toString()) );
        confHandler.setDemoCaptureDelayTime( Integer.parseInt(editCaptureSpeedRealtime.getText().toString()) );
        confHandler.setImageRotation( Integer.parseInt(editPictureRotation.getText().toString()) );
        DeviceConfiguration.getInstance(this).saveConfiguration();
    }



}
