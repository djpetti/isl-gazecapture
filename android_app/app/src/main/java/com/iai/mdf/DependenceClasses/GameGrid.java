package com.iai.mdf.DependenceClasses;

import android.app.Activity;
import android.content.Context;
import android.graphics.drawable.AnimationDrawable;
import android.graphics.drawable.Drawable;
import android.os.CountDownTimer;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;
import android.util.Pair;
import android.widget.ImageButton;
import android.widget.ImageView;

import com.iai.mdf.R;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by mou on 3/24/18.
 */

public class GameGrid {

    private final String LOG_TAG = "GameGrid";
    static final int    MOLE_LIFE_NORMAL = 1500;
    static final int    WRONG_WHACK_PENALTY = 0;
    static final double GAME_SPEED_VARIATION_PROB = 0.90;
    static final int    GAME_MAX_MOLES = 3;

    private Activity ctxt;
    private ArrayList<ImageButton>  holes;
    private ArrayList<ImageView>    holeCovers;
    private ArrayList<Mole> moles;
    private int GRID_SIZE_ROW;
    private int GRID_SIZE_COL;
    private int GAME_SPEED;


    private Handler     gameProcHandler;
    private Runnable    gameGenerateRunnable;

    public GameGrid(Context context, int rowNum, int colNum, int speed){
        ctxt = (Activity) context;
        GRID_SIZE_ROW = rowNum;
        GRID_SIZE_COL = colNum;
        GAME_SPEED = speed;
        holes = new ArrayList<>();
        holeCovers = new ArrayList<>();
        moles = new ArrayList<>();
        gameProcHandler = new Handler();
        gameGenerateRunnable = new Runnable() {
            @Override
            public void run() {
                Mole newMole = generateMole();
                if( newMole!=null ) {
                    showMole(newMole);
                }
                int generator_interval;
                if( new Random().nextFloat()>GAME_SPEED_VARIATION_PROB ){
                    generator_interval = (int)(2-(float)GAME_SPEED/10)*1000;
                } else {
                    generator_interval = (int)(2-(float)GAME_SPEED/10)*1500;
                }
                gameProcHandler.postDelayed(this, generator_interval);
            }
        };
    }



    public void addHole(ImageButton btn, ImageView imageView) {
        this.holes.add(btn);
        this.holeCovers.add(imageView);
    }

    public ImageButton getHole(int idx){
        return holes.get(idx);
    }

    public ArrayList<ImageButton> getHoles(){
        return holes;
    }

    public void startGame(){
        moles.clear();
        gameProcHandler.post(gameGenerateRunnable);

    }

    public void stopGame(){
        gameProcHandler.removeCallbacks(gameGenerateRunnable);
        for(ImageButton eachBtn : holes){
            eachBtn.setImageResource(R.drawable.mole_frame0);
        }
    }

    @Nullable
    private Mole generateMole(){
        if( moles.size() > GAME_MAX_MOLES) {
            return null;
        }
        // generate new valid position  [0, total_unm-1]
        int newPos;
        while(true) {
            newPos = new Random().nextInt(GRID_SIZE_COL*GRID_SIZE_ROW);
            int i;
            for(i = 0 ; i < moles.size(); i++){
                // stop if the id exists
                if( moles.get(i).getId()==newPos ){
                    break;
                }
            }
            // regenerate again if id doesn't exist
            if(i==moles.size()){
                break;
            }
        }
        // equip a mole for the new position
        Mole newMole = new Mole(holes.get(newPos));
        newMole.setGameSpeed(GAME_SPEED);
        newMole.addToGroup(moles);
        return newMole;
    }

    private void showMole(Mole mole){
        mole.startLifeCycle();
    }

    public int  whack(ImageButton btn){
        int earnCash = WRONG_WHACK_PENALTY;
        int cellPos = btn.getId();
        int coverIdx = (cellPos / 10) * GRID_SIZE_COL + cellPos % 10;
        holeCovers.get(coverIdx).setImageResource(R.drawable.anim_hammer);
        AnimationDrawable frameAnimation = (AnimationDrawable) holeCovers.get(coverIdx).getDrawable();
        for (Mole eachMole: moles) {
            if( eachMole.getHoleId()==btn.getId() ){
                earnCash = eachMole.beWhacked(frameAnimation, ctxt);
            }
        }
        frameAnimation.addFrame(ContextCompat.getDrawable(ctxt, android.R.color.transparent), 10);
        frameAnimation.start();
        return earnCash;
    }






    static public class Mole {
        private ImageButton hole;
        private int id;
        private int value;
        private int maxLife;
        private ArrayList<Mole> moleGroup;
        private boolean uiGoingDown;
        private CountDownTimer lifeTimer;

        public Mole(ImageButton imgBtn){
            hole = imgBtn;
            id = imgBtn.getId();
            value = 100;
            uiGoingDown = false;
        }

        public void setGameSpeed(int s){
            maxLife  =  (int)(MOLE_LIFE_NORMAL + 600 * (float)(5-s)/5);
        }

        public void addToGroup(ArrayList<Mole> list){
            moleGroup = list;
            moleGroup.add(this);
        }

        public void startLifeCycle(){
            hole.setImageResource(R.drawable.anim_mole_go_up);
            AnimationDrawable frameAnimation = (AnimationDrawable) hole.getDrawable();
            frameAnimation.start();
            lifeTimer = new CountDownTimer(maxLife, 100){
                @Override
                public void onTick(long millisUntilFinished) {
                    if( maxLife-millisUntilFinished > 300 ) {
                        value = (int) (value * (millisUntilFinished+100) / maxLife);
                    }
                    if( millisUntilFinished<400 && !uiGoingDown ){
                        uiGoingDown = true;
                        hole.setImageResource(R.drawable.anim_mole_go_down);
                        AnimationDrawable frameAnimation = (AnimationDrawable) hole.getDrawable();
                        frameAnimation.start();
                    }
                }

                @Override
                public void onFinish() {
                    value = 0;
                    disappear();
                }
            }.start();
        }

        public void disappear(){
            hole.setImageResource(R.drawable.mole_frame0);
            for(int i = 0; i < moleGroup.size(); ++i){
                if( moleGroup.get(i).getId()==id ){
                    moleGroup.remove(i);
                    break;
                }
            }
        }

        public int beWhacked(AnimationDrawable animationDrawable, Context context){
            if( value>0 ) {
                lifeTimer.cancel();
                id = -1;  // to make the mole invalid, but still in the group
                animationDrawable.addFrame(ContextCompat.getDrawable(context, R.drawable.hammer_frame5_boom), 200);
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        // after a while, finish this mole's life
                        hole.setImageResource(R.drawable.mole_frame0);
                        for(int i = 0; i < moleGroup.size(); ++i){
                            if( moleGroup.get(i).getId()==id ){
                                moleGroup.remove(i);
                                break;
                            }
                        }
                    }
                }, 400);
            }
            return value;
        }

        public int getHoleId(){
            return hole.getId();
        }

        public int getId(){
            return id;
        }



    }


}
