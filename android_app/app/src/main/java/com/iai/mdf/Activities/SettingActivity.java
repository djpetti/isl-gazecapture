package com.iai.mdf.Activities;

import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
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
    private EditText editCalibrationMatrix;
    private Button   btnManuCalibration;
    private Button   btnAutoCalibration;
    private Button   btnResetCalibration;
    private EditText editCalibrationSpeed;
    private EditText editCaptureSpeedCollection;
    private EditText editCaptureSpeedRealtime;
    private EditText editVideoCollectionFPS;
    private EditText editPictureRotation;
    private EditText editDotCandidateRow;
    private EditText editDotCandidateCol;

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

        editCalibrationSpeed = findViewById(R.id.setting_activity_editxt_calibration_speed);
        editCalibrationSpeed.setText(String.valueOf(confHandler.getCalibrationSpeed()));
        editCalibrationMatrix = findViewById(R.id.setting_activity_editxt_calibration_manual);
        float[] matNum = confHandler.getCalibrationMatrix();
        String matStr = "";
        for(int i=0; i < 6; i++){
            String numStr = String.format("%.04f", matNum[i]);
            matStr += numStr + ",";
        }
        matStr = matStr.substring(0, matStr.length()-1);
        editCalibrationMatrix.setText(String.valueOf(matStr));
        btnManuCalibration = findViewById(R.id.setting_activity_btn_item_manu_calibration);
        btnManuCalibration.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Restore preferences
                String matStr = editCalibrationMatrix.getText().toString().replace(" ", "");
                String[] nums = matStr.split(",");
                if( nums.length!=6 ){
                    Toast.makeText(SettingActivity.this, "The matrix needs 6 numbers.", Toast.LENGTH_SHORT).show();
                } else {
                    float[] matNum = new float[]{
                            Float.valueOf(nums[0]),
                            Float.valueOf(nums[1]),
                            Float.valueOf(nums[2]),
                            Float.valueOf(nums[3]),
                            Float.valueOf(nums[4]),
                            Float.valueOf(nums[5])
                    };
                    confHandler.setCalibrationMatrix(matNum);
                    confHandler.saveConfiguration();
                    Toast.makeText(SettingActivity.this,
                                "Calibration is reset: \n" +
                                        nums[0] + ", " + nums[1] + ";\n" +
                                        nums[2] + ", " + nums[3] + ";\n" +
                                        nums[4] + ", " + nums[5],
                            Toast.LENGTH_LONG).show();
                }
            }
        });
        btnAutoCalibration = findViewById(R.id.setting_activity_btn_item_auto_calibration);
        btnAutoCalibration.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Restore preferences
                final SharedPreferences settings = getSharedPreferences(PREFERENCE_NAME, Context.MODE_PRIVATE);
                String lastIp = settings.getString(DemoServerActivity2.BUNDLE_KEY_IP, null);
                String lastPort = settings.getString(DemoServerActivity2.BUNDLE_KEY_PORT, null);
                LayoutInflater inflater = (LayoutInflater) getApplicationContext().getSystemService(LAYOUT_INFLATER_SERVICE);
                View layout = inflater.inflate(R.layout.dialog_server, (ViewGroup) findViewById(R.id.dialog_server_viewgroup));
                final EditText ipInput = layout.findViewById(R.id.dialog_server_editxt_ip);
                final EditText portInput = layout.findViewById(R.id.dialog_server_editxt_port);
                ipInput.setText(lastIp);
                portInput.setText(lastPort);
                new AlertDialog.Builder(SettingActivity.this)
                        .setView(layout)
                        .setTitle("Server Setting")
                        .setPositiveButton("Connect", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int whichButton)
                            {
                                dialog.dismiss();
                                Intent intent = new Intent(SettingActivity.this, CalibrationActivity.class);
                                Bundle extras = new Bundle();
                                extras.putString(DemoServerActivity2.BUNDLE_KEY_IP, ipInput.getText().toString());
                                extras.putInt(DemoServerActivity2.BUNDLE_KEY_PORT, Integer.parseInt(portInput.getText().toString()));
                                intent.putExtras(extras);
                                startActivity(intent);
                                SharedPreferences.Editor editor = settings.edit();
                                editor.putString(DemoServerActivity2.BUNDLE_KEY_IP, ipInput.getText().toString());
                                editor.putString(DemoServerActivity2.BUNDLE_KEY_PORT, portInput.getText().toString());
                                editor.commit();
                            }
                        })
                        .setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int i) {
                                dialog.dismiss();
                            }
                        })
                        .show();
            }
        });
        btnResetCalibration = findViewById(R.id.setting_activity_btn_calibration_reset);
        btnResetCalibration.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                confHandler.setCalibrationMatrix(new float[]{1,0,0,1,0,0});
                confHandler.saveConfiguration();
                Toast.makeText(SettingActivity.this,
                        "Calibration is reset: \n1, 0;\n0, 1;\n0, 0;",
                        Toast.LENGTH_LONG).show();
            }
        });

        editCaptureSpeedCollection = findViewById(R.id.setting_activity_editxt_capture_speed_collection);
        editCaptureSpeedCollection.setText(String.valueOf(confHandler.getCollectionCaptureDelayTime()));
        editCaptureSpeedRealtime = findViewById(R.id.setting_activity_editxt_capture_speed_realtime);
        editCaptureSpeedRealtime.setText(String.valueOf(confHandler.getDemoCaptureDelayTime()));
        editVideoCollectionFPS = findViewById(R.id.setting_activity_editxt_video_collection_fps);
        editVideoCollectionFPS.setText(String.valueOf(confHandler.getVideoCollectionFPS()));

        editPictureRotation = findViewById(R.id.setting_activity_editxt_image_rotation);
        editPictureRotation.setText(String.valueOf(confHandler.getImageRotation()));

        editDotCandidateRow = findViewById(R.id.setting_activity_editxt_dot_candidate_row);
        editDotCandidateRow.setText(String.valueOf(confHandler.getDotCandidateRow()));
        editDotCandidateCol = findViewById(R.id.setting_activity_editxt_dot_candidate_col);
        editDotCandidateCol.setText(String.valueOf(confHandler.getDotCandidateCol()));

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
                btnAutoCalibration.setEnabled(false);
                editMode = !editMode;
                invalidateOptionsMenu();
                return true;

            case R.id.menu_action_save:
                if( isInputValid() ){
                    setEditxtMode(false);
                    btnAutoCalibration.setEnabled(true);
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
        if( editCalibrationSpeed.getText().toString().isEmpty() ){
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
        if( editVideoCollectionFPS.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editPictureRotation.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editDotCandidateRow.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        if( editDotCandidateCol.getText().toString().isEmpty() ){
            Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show();
            return false;
        }
        int degree = Integer.parseInt(editPictureRotation.getText().toString());
        if(  degree!=0 && degree!=90 && degree!=180 && degree!=270 ){
            Toast.makeText(this, "Rotation should be one in {0, 90, 180, 270}", Toast.LENGTH_SHORT).show();
            return false;
        }
        int dotCandRow = Integer.parseInt(editDotCandidateRow.getText().toString());
        if( dotCandRow <= 0 ){
            Toast.makeText(this, "Dot Candidate Row should be an integer greater than 0", Toast.LENGTH_SHORT).show();
            return false;
        }
        int dotCandCol = Integer.parseInt(editDotCandidateCol.getText().toString());
        if( dotCandCol <= 0 ){
            Toast.makeText(this, "Dot Candidate Col should be an integer greater than 0", Toast.LENGTH_SHORT).show();
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
            editCalibrationSpeed.setEnabled(enabled);
            editCaptureSpeedCollection.setEnabled(enabled);
            editCaptureSpeedRealtime.setEnabled(enabled);
            editVideoCollectionFPS.setEnabled(enabled);
            editPictureRotation.setEnabled(enabled);
            editDotCandidateRow.setEnabled(enabled);
            editDotCandidateCol.setEnabled(enabled);
            editCameraPosX.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCameraPosY.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editDisplaySizeX.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editDisplaySizeY.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCalibrationSpeed.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCaptureSpeedCollection.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCaptureSpeedRealtime.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editVideoCollectionFPS.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editPictureRotation.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editDotCandidateRow.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editDotCandidateCol.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            editCameraPosX.setFocusableInTouchMode(enabled);
            editCameraPosY.setFocusableInTouchMode(enabled);
            editDisplaySizeX.setFocusableInTouchMode(enabled);
            editDisplaySizeY.setFocusableInTouchMode(enabled);
            editCalibrationSpeed.setFocusableInTouchMode(enabled);
            editCaptureSpeedCollection.setFocusableInTouchMode(enabled);
            editCaptureSpeedRealtime.setFocusableInTouchMode(enabled);
            editVideoCollectionFPS.setFocusableInTouchMode(enabled);
            editPictureRotation.setFocusableInTouchMode(enabled);
            editDotCandidateRow.setFocusableInTouchMode(enabled);
            editDotCandidateCol.setFocusableInTouchMode(enabled);
        } else {
            editCameraPosX.setEnabled(enabled);
            editCameraPosY.setEnabled(enabled);
            editDisplaySizeX.setEnabled(enabled);
            editDisplaySizeY.setEnabled(enabled);
            editCalibrationSpeed.setEnabled(enabled);
            editCaptureSpeedCollection.setEnabled(enabled);
            editCaptureSpeedRealtime.setEnabled(enabled);
            editVideoCollectionFPS.setEnabled(enabled);
            editPictureRotation.setEnabled(enabled);
            editDotCandidateRow.setEnabled(enabled);
            editDotCandidateCol.setEnabled(enabled);
            editCameraPosX.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCameraPosY.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editDisplaySizeX.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editDisplaySizeY.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCalibrationSpeed.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCaptureSpeedCollection.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCaptureSpeedRealtime.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editVideoCollectionFPS.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editPictureRotation.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editDotCandidateRow.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editDotCandidateCol.setTextColor(ContextCompat.getColor(this, android.R.color.black));
            editCameraPosX.setFocusableInTouchMode(enabled);
            editCameraPosY.setFocusableInTouchMode(enabled);
            editDisplaySizeX.setFocusableInTouchMode(enabled);
            editDisplaySizeY.setFocusableInTouchMode(enabled);
            editCalibrationSpeed.setFocusableInTouchMode(enabled);
            editCaptureSpeedCollection.setFocusableInTouchMode(enabled);
            editCaptureSpeedRealtime.setFocusableInTouchMode(enabled);
            editVideoCollectionFPS.setFocusableInTouchMode(enabled);
            editPictureRotation.setFocusableInTouchMode(enabled);
            editDotCandidateRow.setFocusableInTouchMode(enabled);
            editDotCandidateCol.setFocusableInTouchMode(enabled);
        }
    }


    private void updateSetting(){
        confHandler.setCameraOffsetPWidth( Float.parseFloat(editCameraPosX.getText().toString()) ) ;
        confHandler.setCameraOffsetPHeight( Float.parseFloat(editCameraPosY.getText().toString()) ) ;
        confHandler.setScreenSizePWidth( Float.parseFloat(editDisplaySizeX.getText().toString()) ) ;
        confHandler.setScreenSizePHeight( Float.parseFloat(editDisplaySizeY.getText().toString()) ) ;
        confHandler.setCalibrationSpeed( Integer.parseInt(editCalibrationSpeed.getText().toString()) );
        confHandler.setCollectionCaptureDelayTime( Integer.parseInt(editCaptureSpeedCollection.getText().toString()) );
        confHandler.setDemoCaptureDelayTime( Integer.parseInt(editCaptureSpeedRealtime.getText().toString()) );
        confHandler.setVideoCollectionFPS( Integer.parseInt(editVideoCollectionFPS.getText().toString()) );
        confHandler.setImageRotation( Integer.parseInt(editPictureRotation.getText().toString()) );
        confHandler.setDotCandidateRow( Integer.parseInt(editDotCandidateRow.getText().toString()) );
        confHandler.setDotCandidateCol( Integer.parseInt(editDotCandidateCol.getText().toString()) );
        DeviceConfiguration.getInstance(this).saveConfiguration();
    }



}
