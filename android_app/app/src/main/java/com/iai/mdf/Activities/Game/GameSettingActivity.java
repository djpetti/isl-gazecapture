package com.iai.mdf.Activities.Game;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.Toast;

import com.iai.mdf.Activities.MainActivity;
import com.iai.mdf.R;

import belka.us.androidtoggleswitch.widgets.ToggleSwitch;

/**
 * Created by mou on 3/24/18.
 */

public class GameSettingActivity extends Activity {

    private final String LOG_TAG = "GameSettingActivity";
    public static final String  BUNDLE_KEY_IP = "ip";
    public static final String  BUNDLE_KEY_PORT = "port";



    private SharedPreferences settings;
    private EditText ipInput;
    private EditText portInput;
    private SeekBar seekbarSpeed;
    private CheckBox checkBoxAutoTrigger;
    private Button btnYes;
    private Button btnNo;

    private Button btnClear;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_game_setting);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        hideBottomBar();


        settings = getSharedPreferences(MainActivity.PREFERENCE_NAME, Context.MODE_PRIVATE);
        String lastIp = settings.getString(GameSettingActivity.BUNDLE_KEY_IP, null);
        String lastPort = settings.getString(GameSettingActivity.BUNDLE_KEY_PORT, null);
        ipInput = findViewById(R.id.activity_game_setting_editxt_ip);
        portInput = findViewById(R.id.activity_game_setting_editxt_port);
        ipInput.setText(lastIp);
        portInput.setText(lastPort);

        seekbarSpeed = findViewById(R.id.activity_game_setting_seekbar_speed);

        checkBoxAutoTrigger = findViewById(R.id.activity_game_checkbox_gaze_trigger);
        checkBoxAutoTrigger.setChecked(settings.getBoolean(GameMainActivity.KEY_TRIGGER_MODE, true));


        btnYes = findViewById(R.id.activity_game_setting_btn_confirm);
        btnYes.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // save settings
                SharedPreferences.Editor editor = settings.edit();
                editor.putString(GameSettingActivity.BUNDLE_KEY_IP, ipInput.getText().toString());
                editor.putString(GameSettingActivity.BUNDLE_KEY_PORT, portInput.getText().toString());
                editor.putBoolean(GameMainActivity.KEY_TRIGGER_MODE, checkBoxAutoTrigger.isChecked());
                editor.commit();
                finish();
            }
        });

        btnNo = findViewById(R.id.activity_game_setting_btn_cancel);
        btnNo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

        btnClear = findViewById(R.id.dialog_game_setting_btn_clear_score);
        btnClear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                SharedPreferences.Editor editor = settings.edit();
                editor.putInt(GameMainActivity.KEY_MAX_SCORE, 0);
                editor.commit();
                Toast.makeText(
                        GameSettingActivity.this,
                        "Score record has been reset",
                        Toast.LENGTH_SHORT
                ).show();
            }
        });

    }


    @Override
    protected void onPause() {
        super.onPause();
        overridePendingTransition(0, 0);
    }

    @SuppressLint("NewApi")
    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if(android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT && hasFocus) {
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
        }
    }


    private void hideBottomBar(){
        final int flags = View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
        // This work only for android 4.4+
        if(android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            getWindow().getDecorView().setSystemUiVisibility(flags);
            final View decorView = getWindow().getDecorView();
            decorView.setOnSystemUiVisibilityChangeListener(new View.OnSystemUiVisibilityChangeListener() {
                @Override
                public void onSystemUiVisibilityChange(int visibility) {
                    if((visibility & View.SYSTEM_UI_FLAG_FULLSCREEN) == 0) {
                        decorView.setSystemUiVisibility(flags);
                    }
                }
            });
        } else {
            View v = this.getWindow().getDecorView();
            v.setSystemUiVisibility(View.GONE);
        }
    }


}
