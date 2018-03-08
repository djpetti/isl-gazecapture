package com.iai.mdf.Activities;

import android.os.Bundle;
import android.os.Handler;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentTransaction;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import com.iai.mdf.Fragments.FragmentDataCollection;
import com.iai.mdf.Fragments.FragmentPreview;
import com.iai.mdf.R;

/**
 * Created by mou on 9/16/17.
 */

public class DataCollectionActivity extends AppCompatActivity {

    public static final Size    Image_Size = new Size(480, 640);
    private final String LOG_TAG = "DataCollectionActivity";
    private final String DOT_FRAGMENT_TAG = "DOT_FRAGMENT";

    private boolean doubleBackToExitPressedOnce = false;
    private FragmentManager     fragmentManager;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_data_collection);
        getSupportActionBar().hide();

        fragmentManager = getSupportFragmentManager();
        FragmentPreview fragmentPreview = new FragmentPreview();
        fragmentPreview.setOnActionListener(new FragmentPreview.OnActionListener() {
            @Override
            public void onClick() {
                Log.d(LOG_TAG, "Clicked");
                FragmentDataCollection fragmentDataCollection = new FragmentDataCollection();
                FragmentTransaction transaction = fragmentManager.beginTransaction();
                transaction.replace(R.id.activity_data_collection_layout_fragment_holder, fragmentDataCollection, DOT_FRAGMENT_TAG);
                transaction.commit();
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
        Fragment fragment = fragmentManager.findFragmentByTag(DOT_FRAGMENT_TAG);
        if( null==fragment ){    // if not collecting the training data
            super.onBackPressed();
            return;
        } else {
            if( !((FragmentDataCollection)fragment).isPicSaved() ){
                return;
            }
            if (doubleBackToExitPressedOnce) {
                super.onBackPressed();
                return;
            }
            this.doubleBackToExitPressedOnce = true;
            Toast.makeText(this, "Press Back again to exit", Toast.LENGTH_SHORT).show();
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    doubleBackToExitPressedOnce=false;
                }
            }, 1500);
        }

    }







}
