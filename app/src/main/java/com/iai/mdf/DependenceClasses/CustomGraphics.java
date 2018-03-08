package com.iai.mdf.DependenceClasses;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.Log;
import android.view.View;

/**
 * Created by ShenWang on 11/10/2017.
 */

public class CustomGraphics extends View {

    public static final int TYPE_RECTANGLE = 1;
    public static final int TYPE_CIRCLE = 2;
    private final String LOG_TAG = "CustomGraphics";
    private Paint   paint;
    private int     COLOR = Color.GREEN;
    private int     type;
    private Rect    rectPos;



    public CustomGraphics(Context context, int _type) {
        super(context);
        paint = new Paint();
        paint.setColor(COLOR);
        type = _type;
        rectPos = null;
    }


    public void setColor(int _color){
        COLOR = _color;
        paint.setColor(_color);
    }

    public void setPosition(Rect rect){
        rectPos = rect;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.drawColor(Color.TRANSPARENT);
        switch (type){
            case TYPE_RECTANGLE:
                if(rectPos!=null){
                    canvas.drawRect(rectPos, paint);
                } else {
                    Log.e(LOG_TAG, "The graphics does not show, because rectangle position is not assigned.");
                }
                break;
            default:
                Log.e(LOG_TAG, "The graphics does not show because of an unknown type.");
                break;
        }

    }
}
