package com.iai.mdf.Handlers;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;

/**
 * Created by Mou on 9/28/2017.
 */


public class TensorFlowHandler {

    public static final int MODEL_REGRESSION = 1;
    public static final int MODEL_CLASSIFICATION = 2;
    public static final String MODEL_REGRS_C4_FILE_NAME = "model_regrs4_7500_both_eye_posScaled_7_85.pb";
    public static final String MODEL_REGRS_C6_FILE_NAME = "model_regrs6_7500_both_eye_Pos_Scale_10_0.pb";
    public static final String MODEL_REGRS_C9_FILE_NAME = "model_regrs9_7500_both_eye_Pos_Scale_8_80.pb";
    public static final String MODEL_ISL_FILE_NAME = "model_isl_7500_2.pb";
    public static final String MODEL_CLASS_C4_FILE_NAME = "model_class4_7500_both_eye_posScaled_7_85.pb";
    public static final String MODEL_CLASS_C6_FILE_NAME = "model_class6_7500_both_eye_Pos_Scale_10_0.pb";
    public static final String MODEL_CLASS_C9_FILE_NAME = "model_class9_7500_both_eye_Pos_Scale_8_80.pb";
    public static final String MODEL_ITRACKER_FILE_NAME = "model_iTracker_baseline.pb";
    public static final String MODEL_KANG_FILE_NAME = "head_pose_aug_v1.pb";



    private final String LOG_TAG = "TensorFlowHandler";

    private static TensorFlowHandler myInstance;
    private TensorFlowInferenceInterface tf;
    private static final String MODEL_FILE = "file:///android_asset/model_classification4.pb";
    private static final String INPUT_NODE =  "eye_1";
    private static final String[] OUTPUT_NODES = {"head_pose_aug0"};
    private static AssetManager assetManager;

    public static TensorFlowHandler getInstance(final Context context) {
        if (myInstance == null)
        {
            myInstance = new TensorFlowHandler(context);
        }
        return myInstance;
    }

    public TensorFlowHandler(final Context context) {
        this.assetManager = context.getAssets();
        tf = new TensorFlowInferenceInterface(assetManager, MODEL_CLASS_C4_FILE_NAME);
    }

    public void pickModel(String modelName){
        tf = new TensorFlowInferenceInterface(assetManager, modelName);
    }


    public float[] getClassificationResult( ArrayList<String> inputNodes, ArrayList<float[]> inputs, ArrayList<int[]> inputSizes){
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        for (int i=0; i < inputNodes.size(); ++i) {
            int[] inputSize = inputSizes.get(i);
            switch (inputSize.length ){
                case 2:
                    tf.feed(inputNodes.get(i), inputs.get(i), 1, inputSize[0], inputSize[1]);
                    break;
                case 3:
                    tf.feed(inputNodes.get(i), inputs.get(i), 1, inputSize[0], inputSize[1], inputSize[2]);
                    break;
                case 4:
                    tf.feed(inputNodes.get(i), inputs.get(i), 1, inputSize[0], inputSize[1], inputSize[2], inputSize[3]);
                    break;
                default:
                    return null;
            }
        }
        Trace.endSection();
        // Run the inference call.
        Trace.beginSection("run");
        tf.run(OUTPUT_NODES , false);
        Trace.endSection();
        // Copy the output Tensor back into the output array.
        final Operation operation = tf.graphOperation(OUTPUT_NODES[0]);
        final int classNum = (int) operation.output(0).shape().size(1);
        float[] output = new float[classNum];
        Trace.beginSection("fetch");
        tf.fetch(OUTPUT_NODES[0], output);
        Trace.endSection();
        return output;
    }


    public float[] getEstimatedLocation(ArrayList<String> inputNodes, ArrayList<float[]> inputs, ArrayList<int[]> inputSizes){
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        for (int i=0; i < inputNodes.size(); ++i) {
            int[] inputSize = inputSizes.get(i);
            switch (inputSize.length ){
                case 1:
                    tf.feed(inputNodes.get(i), inputs.get(i), 1, inputSize[0]);
                    break;
                case 2:
                    tf.feed(inputNodes.get(i), inputs.get(i), 1, inputSize[0], inputSize[1]);
                    break;
                case 3:
                    tf.feed(inputNodes.get(i), inputs.get(i), 1, inputSize[0], inputSize[1], inputSize[2]);
                    break;
                case 4:
                    tf.feed(inputNodes.get(i), inputs.get(i), 1, inputSize[0], inputSize[1], inputSize[2], inputSize[3]);
                    break;
                default:
                    return null;
            }
        }
        Trace.endSection();
        // Run the inference call.
        Trace.beginSection("run");
        tf.run(OUTPUT_NODES , false);
        Trace.endSection();
        // Copy the output Tensor back into the output array.
        float[] output = new float[3];
        Trace.beginSection("fetch");
        tf.fetch(OUTPUT_NODES[0], output);
        Trace.endSection();
        return output;
    }


    public float[] iThackerCM2Loc(float[] cmCoor){
        float[] relLocInPortrait = new float[2];
        relLocInPortrait[0] = (float) ((cmCoor[1] + 1.85) / 6.25);
        relLocInPortrait[1] = (float) ((cmCoor[0] - 1.05) / 11.05);
        return relLocInPortrait;
    }

//    public TensorFlowHandler(final Context context) {
//        this.assetManager = context.getAssets();
//        tf = new TensorFlowInferenceInterface();
//        if( 0 != tf.initializeTensorFlow(assetManager, MODEL_FILE_LOCAL) ){
//            throw new RuntimeException("TensorFlow initialization Fails in TensorFlowHandler.Java");
//        }
//    }
//
//    public float[] getTestLocation(float[] array){
//        tf.fillNodeFloat(INPUT_NODE, new int[]{1, 36, 60, 3}, array);
//        tf.runInference(OUTPUT_NODES);
//        // retrieve result from TensorFlow model
//        float[] loc = new float[]{0,0};
//        tf.readNodeFloat(OUTPUT_NODES[0], loc);
////        Point res = new Point();
////        res.set(  loc[0],loc[1] );
//        return loc;
//    }
//
//    public float[] getTestLabel(float[] array){
//        tf.fillNodeFloat(INPUT_NODE, new int[]{1, 36, 60, 3}, array);
//        tf.runInference(OUTPUT_NODES);
//        // retrieve result from TensorFlow model
//        float[] loc = new float[]{0,0,0,0};
//        tf.readNodeFloat(OUTPUT_NODES[0], loc);
////        int label = 0;
////        if( loc[0] > 0.5 && loc[1] < 0.5 ){
////            label = 1;
////        } else if( loc[0] > 0.5 && loc[1] >= 0.5 ) {
////            label = 2;
////        } else if( loc[0] <= 0.5 && loc[1] < 0.5 ) {
////            label = 3;
////        } else {
////            label = 4;
////        }
//        return loc;
//    }
//
//    public float[] getEstimatedLocation(float[][][][] images){
//        float[] input_array = ImageProcessHandler.toTensorFlowEyeModelArray(images);
//        tf.fillNodeFloat(
//                INPUT_NODE,
//                new int[]{
//                        images.length,
//                        ImageProcessHandler.EYE_MODEL_INPUTSIZE_ROWS,
//                        ImageProcessHandler.EYE_MODEL_INPUTSIZE_COLUMNS,
//                        ImageProcessHandler.EYE_MODEL_INPUTSIZE_COLORS},
//                input_array
//        );
//        tf.runInference(OUTPUT_NODES);
//        // retrieve result from TensorFlow model
//        float[] loc = new float[ images.length * 2];
//        tf.readNodeFloat(OUTPUT_NODES[0], loc);
//        return loc;
//    }
//
//    public float[] getResultFromNewPbFile(float[][][] images){
//        float[] input_array = ImageProcessHandler.toTensorFlowEyeModelArray(images);
//        tf.fillNodeFloat(INPUT_NODE,
//                new int[]{images.length, images[0].length, images[0][0].length, 1},
//                input_array);
//        tf.runInference(OUTPUT_NODES);
//        // retrieve result from TensorFlow model
//        float[] loc = new float[ images.length * 2];
//        tf.readNodeFloat(OUTPUT_NODES[0], loc);
//        return loc;
//    }



}
