package com.iai.mdf.Activities;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;

import com.iai.mdf.Activities.Game.GameMenuActivity;
import com.iai.mdf.DependenceClasses.DeviceConfiguration;
import com.iai.mdf.R;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {


    public static final int MY_PERMISSIONS_REQUEST_ACCESS_CODE = 1;
    public static final String PREFERENCE_NAME = "isl_mobile_eye_gaze";
    private static final String LOG_TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getSupportActionBar().hide();
        checkPermissions();
        DeviceConfiguration.getInstance(this).loadConfiguration();


        Button btn_data_collection =  (Button) findViewById(R.id.main_activity_btn_data_collection);
        btn_data_collection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, DataCollectionActivity.class);
                startActivity(intent);
            }
        });
//        Button btn_data_process = (Button) findViewById(R.id.main_activity_btn_data_process);
//        btn_data_process.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                Intent intent = new Intent(MainActivity.this, DataProcessActivity.class);
//                startActivity(intent);
//            }
//        });
        Button btn_data_process = (Button) findViewById(R.id.main_activity_btn_game);
        btn_data_process.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, GameMenuActivity.class);
                startActivity(intent);
            }
        });
        Button btn_demo_class = (Button) findViewById(R.id.main_activity_btn_classification);
        btn_demo_class.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, DemoClassActivity.class);
                startActivity(intent);
            }
        });
        Button btn_demo_regression = (Button) findViewById(R.id.main_activity_btn_regression);
        btn_demo_regression.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, DemoRegrsActivity.class);
                startActivity(intent);
            }
        });
        Button btn_demo_iTracker = (Button) findViewById(R.id.main_activity_btn_iTracker);
        btn_demo_iTracker.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, DemoiTrackerActivity.class);
                startActivity(intent);
            }
        });
//        Button btn_tensorflow_temp= (Button) findViewById(R.id.main_activity_btn_tensorflow_speed_test);
//        btn_tensorflow_temp.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                Intent intent = new Intent(MainActivity.this, CloudActivity.class);
//                startActivity(intent);
//            }
//        });
        Button btn_isl_server2 = (Button) findViewById(R.id.main_activity_btn_tensorflow_isl_server2);
        btn_isl_server2.setOnClickListener(new View.OnClickListener() {
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
                new AlertDialog.Builder(MainActivity.this)
                        .setView(layout)
                        .setTitle("Server Setting")
                        .setPositiveButton("Connect", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int whichButton)
                            {
                                dialog.dismiss();
                                Intent intent = new Intent(MainActivity.this, DemoServerActivity2.class);
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
        Button btn_isl_server1 = (Button) findViewById(R.id.main_activity_btn_tensorflow_isl_server1);
        btn_isl_server1.setOnClickListener(new View.OnClickListener() {
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
                new AlertDialog.Builder(MainActivity.this)
                        .setView(layout)
                        .setTitle("Server Setting")
                        .setPositiveButton("Connect", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int whichButton)
                            {
                                dialog.dismiss();
                                Intent intent = new Intent(MainActivity.this, DemoServerActivity1.class);
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
        Button btn_test_pb = findViewById(R.id.main_activity_btn_tensorflow_pb_test);
        btn_test_pb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, TestPBActivity.class);
                startActivity(intent);
            }
        });
        Button btn_setting = findViewById(R.id.main_activity_btn_setting);
        btn_setting.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, SettingActivity.class);
                startActivity(intent);
            }
        });
        Button btn_exit =  findViewById(R.id.main_activity_btn_exit);
        btn_exit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

    }




    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String permissions[], @NonNull int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_ACCESS_CODE: {
                if (!(grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    checkPermissions();
                }
            }
        }
    }

    /**
     * checking  permissions at Runtime.
     */
    @TargetApi(Build.VERSION_CODES.M)
    private void checkPermissions() {
        final String[] requiredPermissions = {
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA
        };
        final List<String> neededPermissions = new ArrayList<>();
        for (final String permission : requiredPermissions) {
            if (ContextCompat.checkSelfPermission(getApplicationContext(),
                    permission) != PackageManager.PERMISSION_GRANTED) {
                neededPermissions.add(permission);
            }
        }
        if (!neededPermissions.isEmpty()) {
            requestPermissions(neededPermissions.toArray(new String[]{}),
                    MY_PERMISSIONS_REQUEST_ACCESS_CODE);
        }
    }






}
