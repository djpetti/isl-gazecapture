package com.iai.mdf.Activities;

import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.os.Handler;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentTransaction;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import com.iai.mdf.Fragments.FragmentDataCollectionByPicture;
import com.iai.mdf.Fragments.FragmentDataCollectionByVideo;
import com.iai.mdf.Fragments.FragmentPreview;
import com.iai.mdf.R;

/**
 * Created by mou on 9/16/17.
 */

public class DataCollectionActivity extends AppCompatActivity {

    public static final Size    Image_Size = new Size(480, 640);
    private final String LOG_TAG = "DataCollectionActivity";
    private final String FRAGMENT_TAG_PICTURE = "PICTURE_FRAGMENT";
    private final String FRAGMENT_TAG_VIDEO = "VIDEO_FRAGMENT";

    private boolean doubleBackToExitPressedOnce = false;
    private FragmentManager     fragmentManager;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_data_collection);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getSupportActionBar().hide();

        fragmentManager = getSupportFragmentManager();
        FragmentPreview fragmentPreview = new FragmentPreview();
//        fragmentPreview.setOnActionListener(new FragmentPreview.OnActionListener() {
//            @Override
//            public void onClick() {
//                Log.d(LOG_TAG, "Clicked");
//                FragmentDataCollectionByPicture fragmentDataCollectionByPicture = new FragmentDataCollectionByPicture();
//                FragmentTransaction transaction = fragmentManager.beginTransaction();
//                transaction.replace(R.id.activity_data_collection_layout_fragment_holder, fragmentDataCollectionByPicture, FRAGMENT_TAG_PICTURE);
//                transaction.commit();
//            }
//        });
        fragmentPreview.setButtonClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                FragmentTransaction transaction = fragmentManager.beginTransaction();
                switch (v.getId()) {
                    case R.id.fragment_preview_btn_picture:
                        FragmentDataCollectionByPicture fragmentDataCollectionByPicture = new FragmentDataCollectionByPicture();
                        transaction.replace(R.id.activity_data_collection_layout_fragment_holder, fragmentDataCollectionByPicture, FRAGMENT_TAG_PICTURE);
                        transaction.commit();
                        break;

                    case R.id.fragment_preview_btn_video:
                        FragmentDataCollectionByVideo fragmentDataCollectionByVideo = new FragmentDataCollectionByVideo();
                        transaction.replace(R.id.activity_data_collection_layout_fragment_holder, fragmentDataCollectionByVideo, FRAGMENT_TAG_VIDEO);
                        transaction.commit();
                        break;
                }
            }
        });
        fragmentManager
                .beginTransaction()
                .add(R.id.activity_data_collection_layout_fragment_holder, fragmentPreview)
                .commit();

    }




    /**
     * Press Back twice to exit unless a picture is being
     * saved. If so, try again after the picture is saved
     */
    @Override
    public void onBackPressed() {
        Fragment fragment = fragmentManager.findFragmentById(R.id.activity_data_collection_layout_fragment_holder);
        if( fragment instanceof FragmentPreview ){    // if not collecting the training data
            super.onBackPressed();
            return;
        } else if( fragment instanceof FragmentDataCollectionByPicture ) {
            if( !((FragmentDataCollectionByPicture)fragment).isPicSaved() ){
                return;
            }
            if (doubleBackToExitPressedOnce) {
                super.onBackPressed();
                return;
            }
            this.doubleBackToExitPressedOnce = true;
            Toast.makeText(this, "Press Back twice to exit", Toast.LENGTH_SHORT).show();
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    doubleBackToExitPressedOnce=false;
                }
            }, 1200);
        } else if ( fragment instanceof FragmentDataCollectionByVideo ) {
            if (doubleBackToExitPressedOnce) {
                super.onBackPressed();
                return;
            }
            this.doubleBackToExitPressedOnce = true;
            Toast.makeText(this, "Press Back twice to exit", Toast.LENGTH_SHORT).show();
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    doubleBackToExitPressedOnce=false;
                }
            }, 1200);
        }

    }







}
