package com.iai.mdf.Activities.Game;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.Dialog;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.SeekBar;

import com.iai.mdf.Activities.DemoServerActivity2;
import com.iai.mdf.Activities.MainActivity;
import com.iai.mdf.R;


/**
 * Created by mou on 3/23/18.
 */

public class GameMenuActivity extends Activity {

    private final String LOG_TAG = "GameMenuActivity";



    private Button btnStart33;
    private Button btnStart34;
    private ImageButton  btnSetting;
    private ImageButton  btnExit;



    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_game_menu);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        hideBottomBar();


        btnStart33 = findViewById(R.id.activity_game_mole_btn_start_33);
        btnStart33.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(LOG_TAG, "Game Begin");
                Intent intent = new Intent(GameMenuActivity.this, GameMainActivity.class);
                Bundle extras = new Bundle();
                extras.putInt(GameMainActivity.KEY_GRID_SIZE, 33);
                extras.putInt(GameMainActivity.KEY_MODE, GameMainActivity.VALUE_MODE_TIMER);
                intent.putExtras(extras);
                overridePendingTransition(0, 0);
                startActivity(intent);
            }
        });

        btnStart34 = findViewById(R.id.activity_game_mole_btn_start_34);
        btnStart34.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(LOG_TAG, "Game Begin");
                Intent intent = new Intent(GameMenuActivity.this, GameMainActivity.class);
                Bundle extras = new Bundle();
                extras.putInt(GameMainActivity.KEY_GRID_SIZE, 34);
                extras.putInt(GameMainActivity.KEY_MODE, GameMainActivity.VALUE_MODE_TIMER);
                intent.putExtras(extras);
                overridePendingTransition(0, 0);
                startActivity(intent);
            }
        });


        btnSetting = findViewById(R.id.activity_game_mole_btn_setting);
        btnSetting.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(GameMenuActivity.this, GameSettingActivity.class);
                overridePendingTransition(0, 0);
                startActivity(intent);
            }
        });

        btnExit = findViewById(R.id.activity_game_mole_btn_exit);
        btnExit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });


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
