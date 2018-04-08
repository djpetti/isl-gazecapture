package com.iai.mdf.Handlers;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Point;
import android.media.Image;
import android.os.Environment;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

import com.iai.mdf.Activities.MainActivity;
import com.iai.mdf.JNInterface.MobileGazeJniInterface;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Created by Mou on 10/1/2017.
 */

public class ImageProcessHandler {

    private final String LOG_TAG = "ImageProcessHandler";
    public static int EYE_MODEL_INPUTSIZE_ROWS      = 36;
    public static int EYE_MODEL_INPUTSIZE_COLUMNS   = 60;
    public static int EYE_MODEL_INPUTSIZE_COLORS    = 3;

    private static MobileGazeJniInterface jniHandler = new MobileGazeJniInterface();


    public static float[][][][] getStandardizedNormalDistribution(int[][][][] rawImages){
        int dim0 = rawImages.length;
        int dim1 = rawImages[0].length;
        int dim2 = rawImages[0][0].length;
        int dim3 = rawImages[0][0][0].length;
        float[][][][] postImage = new float[dim0][dim1][dim2][dim3];
        float[] postArray = new float[dim0*dim1*dim2*dim3];
        int[] eachPreArray = new int[dim1*dim2*dim3];
        for(int d0 = 0; d0 < dim0; ++d0) {
            // 3D array --> row array
            for (int d1 = 0; d1 < dim1; ++d1) {
                for (int d2 = 0; d2 < dim2; ++d2) {
                    for (int d3 = 0; d3 < dim3; ++d3) {
                        eachPreArray[d1 * dim2 * dim3 + d2 * dim3 + d3] = rawImages[d0][d1][d2][d3];
                    }
                }
            }
            // calculate
            double mean = 0;
            for(int i = 0; i < eachPreArray.length; i++) {
                mean += eachPreArray[i];
            }
            mean /= eachPreArray.length;
            double stdev = 0;
            for(int i = 0; i < eachPreArray.length; i++) {
                stdev += (eachPreArray[i] - mean) * (eachPreArray[i] - mean);
            }
            stdev = Math.sqrt( stdev/(eachPreArray.length-1) );
            // preArray --> postArray
            for(int i = 0; i < eachPreArray.length; i++) {
                postArray[d0 * dim1 * dim2 * dim3 + i] = (float)((eachPreArray[i] - mean)/stdev);
            }
            // postArray --> postImage
            for (int d1 = 0; d1 < dim1; ++d1) {
                for (int d2 = 0; d2 < dim2; ++d2) {
                    for (int d3 = 0; d3 < dim3; ++d3) {
                        postImage[d0][d1][d2][d3] = postArray[d0 * dim1 * dim2 * dim3 + d1 * dim2 * dim3 + d2 * dim3 + d3];
                    }
                }
            }
        }
        return postImage;
    }

    public static float[][][] getStandardizedNormalDistribution(int[][][] rawImages){
        int dim0 = rawImages.length;
        int dim1 = rawImages[0].length;
        int dim2 = rawImages[0][0].length;
        float[][][] postImage = new float[dim0][dim1][dim2];
        float[] postArray = new float[dim0*dim1*dim2];
        int[] eachPreArray = new int[dim1*dim2];
        for(int d0 = 0; d0 < dim0; ++d0) {
            if( rawImages[d0][0][0]==0 ){
                for (int d1 = 0; d1 < dim1; ++d1) {
                    for (int d2 = 0; d2 < dim2; ++d2) {
                        postImage[d0][d1][d2] = rawImages[d0][d1][d2];
                    }
                }
                continue;
            }
            // 3D array --> row array
            for (int d1 = 0; d1 < dim1; ++d1) {
                for (int d2 = 0; d2 < dim2; ++d2) {
                    eachPreArray[d1 * dim2 + d2] = rawImages[d0][d1][d2];
                }
            }
            // calculate
            double mean = 0;
            for(int i = 0; i < eachPreArray.length; i++) {
                mean += eachPreArray[i];
            }
            mean /= eachPreArray.length;
            double stdev = 0;
            for(int i = 0; i < eachPreArray.length; i++) {
                stdev += (eachPreArray[i] - mean) * (eachPreArray[i] - mean);
            }
            stdev = Math.sqrt( stdev/(eachPreArray.length-1) );
            // preArray --> postArray
            for(int i = 0; i < eachPreArray.length; i++) {
                postArray[d0 * dim1 * dim2 + i] = (float)((eachPreArray[i] - mean)/stdev);
            }
            // postArray --> postImage
            for (int d1 = 0; d1 < dim1; ++d1) {
                for (int d2 = 0; d2 < dim2; ++d2) {
                    postImage[d0][d1][d2] = postArray[d0 * dim1 * dim2 + d1 * dim2 + d2];
                }
            }
        }
        return postImage;
    }

    public static float[] toTensorFlowEyeModelArray(float[][][][] image){
        int pic_num = image.length;
        float[] input_array = new float[ pic_num * EYE_MODEL_INPUTSIZE_ROWS * EYE_MODEL_INPUTSIZE_COLUMNS * EYE_MODEL_INPUTSIZE_COLORS];
        for(int pic = 0; pic < pic_num; ++pic) {
            for (int row = 0; row < EYE_MODEL_INPUTSIZE_ROWS; ++row) {
                for (int col = 0; col < EYE_MODEL_INPUTSIZE_COLUMNS; ++col) {
                    for (int colorLayer = 0; colorLayer < EYE_MODEL_INPUTSIZE_COLORS; ++colorLayer) {
                        input_array[    pic * EYE_MODEL_INPUTSIZE_ROWS * EYE_MODEL_INPUTSIZE_COLUMNS * EYE_MODEL_INPUTSIZE_COLORS
                                        + row * EYE_MODEL_INPUTSIZE_COLUMNS * EYE_MODEL_INPUTSIZE_COLORS
                                        + col * EYE_MODEL_INPUTSIZE_COLORS
                                        + colorLayer ] = image[pic][row][col][colorLayer];
                    }
                }
            }
        }
        return input_array;
    }

    public static float[] toTensorFlowEyeModelArray(float[][][] image){
        int pic_num = image.length;
        int total_row = image[0].length;
        int total_col = image[0][0].length;
        float[] input_array = new float[ pic_num * total_col * total_row ];
        for(int pic = 0; pic < pic_num; ++pic) {
            for (int row = 0; row < total_row; ++row) {
                for (int col = 0; col < total_col; ++col) {
                    input_array[  pic * total_row * total_col
                            + row * total_col
                            + col ] = image[pic][row][col];
                }
            }
        }
        return input_array;
    }

    public static Mat getBGRMatFromImage(Image image) {
        ByteBuffer buffer;
        int rowStride;
        int pixelStride;
        int width = image.getWidth();
        int height = image.getHeight();
        int offset = 0;

        Image.Plane[] planes = image.getPlanes();
        byte[] data = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];

        for (int i = 0; i < planes.length; i++) {
            buffer = planes[i].getBuffer();
            rowStride = planes[i].getRowStride();
            pixelStride = planes[i].getPixelStride();
            int w = (i == 0) ? width : width / 2;
            int h = (i == 0) ? height : height / 2;
            for (int row = 0; row < h; row++) {
                int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
                if (pixelStride == bytesPerPixel) {
                    int length = w * bytesPerPixel;
                    buffer.get(data, offset, length);

                    if (h - row != 1) {
                        buffer.position(buffer.position() + rowStride - length);
                    }
                    offset += length;
                } else {
                    if (h - row == 1) {
                        buffer.get(rowData, 0, width - pixelStride + 1);
                    } else {
                        buffer.get(rowData, 0, rowStride);
                    }
                    for (int col = 0; col < w; col++) {
                        data[offset++] = rowData[col * pixelStride];
                    }
                }
            }
        }

        Mat mat = new Mat(height + height / 2, width, CvType.CV_8UC1);
        mat.put(0, 0, data);

        return mat;
    }

    public static void getRGBMat(Image image, long addrMat){
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
        int yLength = yBuffer.capacity();
        int uvLength = uBuffer.capacity();
        byte[] yuvBytes = new byte[yLength+2*uvLength];
        yBuffer.get(yuvBytes, 0, yLength);
        uBuffer.get(yuvBytes, yLength, uvLength);
        vBuffer.get(yuvBytes, yLength + uvLength, uvLength);
        int width = image.getWidth();
        int height = image.getHeight();
        jniHandler.getRGBMatImage(yuvBytes, width, height, addrMat);
    }

    public static float[] getEyePostion(double[] landmarks, boolean isLeft, float[] scales){
        float widthScale = scales[0];
        float heightScale = scales[1];
        int leftEyeLeftCornerIndex = 28;
        int leftEyeRightCornerIndex = 25;
        int rightEyeLeftCornerIndex = 22;
        int rightEyeRightCornerIndex = 19;
        Point leftCorner;
        Point rightCorner;
        if( isLeft ) {
            leftCorner = new Point(
                    (int) landmarks[leftEyeLeftCornerIndex * 2],
                    (int) landmarks[leftEyeLeftCornerIndex * 2 + 1]);
            rightCorner = new Point(
                    (int) landmarks[leftEyeRightCornerIndex * 2],
                    (int) landmarks[leftEyeRightCornerIndex * 2 + 1]);
        } else {
            leftCorner = new Point(
                    (int) landmarks[rightEyeLeftCornerIndex * 2],
                    (int) landmarks[rightEyeLeftCornerIndex * 2 + 1]);
            rightCorner = new Point(
                    (int) landmarks[rightEyeRightCornerIndex * 2],
                    (int) landmarks[rightEyeRightCornerIndex * 2 + 1]);
        }
        Point center = new Point(
                (leftCorner.x+rightCorner.x)/2,
                (leftCorner.y+rightCorner.y)/2);
        float width = Math.abs(leftCorner.x - rightCorner.x);
        float height = Math.abs(leftCorner.y - rightCorner.y);
        return new float[]{center.x/640, center.y/480, width/640*widthScale, height/480*heightScale};
    }

    public static int[] getEyeRegionCropRect(double[] landmarks, int imageWidth, int imageHeight, boolean isLeft){
        int[] cropRect = null;
        int leftEyeLeftCornerIndex = 28;
        int leftEyeRightCornerIndex = 25;
        int rightEyeLeftCornerIndex = 22;
        int rightEyeRightCornerIndex = 19;
        Point leftCorner;
        Point rightCorner;
        if( isLeft ) {
            leftCorner = new Point(
                    (int) landmarks[leftEyeLeftCornerIndex * 2],
                    (int) landmarks[leftEyeLeftCornerIndex * 2 + 1]);
            rightCorner = new Point(
                    (int) landmarks[leftEyeRightCornerIndex * 2],
                    (int) landmarks[leftEyeRightCornerIndex * 2 + 1]);
        } else {
            leftCorner = new Point(
                    (int) landmarks[rightEyeLeftCornerIndex * 2],
                    (int) landmarks[rightEyeLeftCornerIndex * 2 + 1]);
            rightCorner = new Point(
                    (int) landmarks[rightEyeRightCornerIndex * 2],
                    (int) landmarks[rightEyeRightCornerIndex * 2 + 1]);
        }

        Point center = new Point(
                (leftCorner.x+rightCorner.x)/2,
                (leftCorner.y+rightCorner.y)/2);
        int width = Math.abs(leftCorner.x - rightCorner.x);
        double padRatio = 0.25;
        int newWidth = (int) (width * (1 + padRatio * 2));
        int newHeight = newWidth * ImageProcessHandler.EYE_MODEL_INPUTSIZE_ROWS / ImageProcessHandler.EYE_MODEL_INPUTSIZE_COLUMNS;
        int newX = center.x - newWidth / 2;
        int newY = center.y - newHeight / 2;
        if( newX >= 0 && newX + newWidth < imageWidth
                && newY >= 0 && newY + newHeight < imageHeight){
            cropRect = new int[4];
            cropRect[0] = newX;
            cropRect[1] = newY;
            cropRect[2] = newWidth;
            cropRect[3] = newHeight;
        }
        return cropRect;
    }

    public static int[] getEyeRegionCropRectForiTracker(double[] landmarks, int imageWidth, int imageHeight, boolean isLeft){
        int[] cropRect = null;
        int leftEyeLeftCornerIndex = 28;
        int leftEyeRightCornerIndex = 25;
        int rightEyeLeftCornerIndex = 22;
        int rightEyeRightCornerIndex = 19;
        Point leftCorner;
        Point rightCorner;
        if( isLeft ) {
            leftCorner = new Point(
                    (int) landmarks[leftEyeLeftCornerIndex * 2],
                    (int) landmarks[leftEyeLeftCornerIndex * 2 + 1]);
            rightCorner = new Point(
                    (int) landmarks[leftEyeRightCornerIndex * 2],
                    (int) landmarks[leftEyeRightCornerIndex * 2 + 1]);
        } else {
            leftCorner = new Point(
                    (int) landmarks[rightEyeLeftCornerIndex * 2],
                    (int) landmarks[rightEyeLeftCornerIndex * 2 + 1]);
            rightCorner = new Point(
                    (int) landmarks[rightEyeRightCornerIndex * 2],
                    (int) landmarks[rightEyeRightCornerIndex * 2 + 1]);
        }

        Point center = new Point(
                (leftCorner.x+rightCorner.x)/2,
                (leftCorner.y+rightCorner.y)/2);
        int width = Math.abs(leftCorner.x - rightCorner.x);
        double padRatio = 0.5;
        int newWidth = (int) (width * (1 + padRatio * 2));
        int newHeight = newWidth;
        int newX = center.x - newWidth / 2;
        int newY = center.y - newHeight / 2;
        if( newX >= 0 && newX + newWidth < imageWidth
                && newY >= 0 && newY + newHeight < imageHeight){
            cropRect = new int[4];
            cropRect[0] = newX;
            cropRect[1] = newY;
            cropRect[2] = newWidth;
            cropRect[3] = newHeight;
        }
        return cropRect;
    }

    // resize[0] is width; resize[1] is height
    public static void cropSingleRegion(long addrMat, int[] rect, int[] resize, float[] tensorflowInput, long cropMatAddr){
        jniHandler.cropImage(addrMat, rect, resize, tensorflowInput, cropMatAddr);
    }

    public static void cropSingleRegionAndSaveTFInput(long addrMat, int[] rect, int[] size, float[] tensorflowInput, long cropMatAddr, String path){
        jniHandler.cropImageAndSaveInput(addrMat, rect, size, tensorflowInput, cropMatAddr, path);
    }

    public static boolean rotateImage(Mat img, int degree){
        if( degree == Core.ROTATE_90_CLOCKWISE
                || degree == Core.ROTATE_180
                || degree == Core.ROTATE_90_COUNTERCLOCKWISE ) {
            jniHandler.rotateImage(
                    img.getNativeObjAddr(),
                    degree);
            return true;
        } else {
            return false;
        }
    }

    public static int[] getRotatedRGBImage(Image image){
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
        byte[] yBytes = new byte[yBuffer.capacity()];
        byte[] uBytes = new byte[uBuffer.capacity()];
        byte[] vBytes = new byte[vBuffer.capacity()];
        yBuffer.get(yBytes);
        uBuffer.get(uBytes);
        vBuffer.get(vBytes);
        int width = image.getWidth();
        int height = image.getHeight();
        MobileGazeJniInterface jniHandler = new MobileGazeJniInterface();
        int[] rgbIntArray = jniHandler.getRotatedRGBImage(yBytes, uBytes, vBytes, width, height);
        return rgbIntArray;
    }

    public static void doFaceLandmarkDetection(long addrMat){
        FaceLandmark detectRes = new FaceLandmark();
    }

    public static float[] faceRectToGridArray(int[] face, int[] gridSize){
        float[] grid = new float[gridSize[0]*gridSize[1]];
        int faceX = face[0] * gridSize[1] / 640;
        int faceY = face[1] * gridSize[0] / 480;
        int faceW = face[2] * gridSize[1] / 640;
        int faceH = face[3] * gridSize[0] / 480;
        for(int c = faceX; c < faceX+faceW; c++){
            for(int r = faceY; r < faceY+faceH; r++){
                grid[gridSize[1]*r + c] = 1;
            }
        }
        return grid;
    }

    public static float[] eyeGrid(int[] leftEye, int[] rightEye, int[] gridSize){
        // girdSize = [grid_height, grid_width]
        float[] grid = new float[gridSize[0]*gridSize[1]];
        int leftEyeX = leftEye[0] * gridSize[1] / 640;
        int leftEyeY = leftEye[1] * gridSize[0] / 480;
        int leftEyeW = leftEye[2] * gridSize[1] / 640;
        int leftEyeH = leftEye[3] * gridSize[0] / 480;
        for(int c = leftEyeX; c < leftEyeX+leftEyeW; c++){
            for(int r = leftEyeY; r < leftEyeY+leftEyeH; r++){
                grid[gridSize[0]*c + r] = 1;
            }
        }
        int rightEyeX = rightEye[0] * gridSize[1] / 640;
        int rightEyeY = rightEye[1] * gridSize[0] / 480;
        int rightEyeW = rightEye[2] * gridSize[1] / 640;
        int rightEyeH = rightEye[3] * gridSize[0] / 480;
        for(int c = rightEyeX; c < rightEyeX+rightEyeW; c++){
            for(int r = rightEyeY; r < rightEyeY+rightEyeH; r++){
                grid[gridSize[0]*c + r] = 1;
            }
        }
        return grid;
    }

    public static float[] standardizeGridArray(float[] grid, int[] gridSize){
        // girdSize = [grid_height, grid_width]
        float[] arr = new float[gridSize[0]*gridSize[1]];
        // normalize
        float sum = 0;
        for(float each : grid){
            sum += each;
        }
        float mean = sum / gridSize[0] / gridSize[1];
        // stddev
        float dev = 0;
        for(float each : grid){
            dev += (each - mean) * (each - mean);
        }
        dev =  (float)Math.sqrt((double)dev/(gridSize[0]*gridSize[1]-1));
        // standardize
        for(int i=0; i < gridSize[0]*gridSize[1]; i++){
            arr[i] = (grid[i] - mean) / dev;
        }
        return arr;
    }

    public static void doFaceEyeDetection(Image image, double[] faces, double[] eyes, float[] eyeRegion){
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
        int yLength = yBuffer.capacity();
        int uvLength = uBuffer.capacity();
        byte[] yuvBytes = new byte[yLength+2*uvLength];
        yBuffer.get(yuvBytes, 0, yLength);
        uBuffer.get(yuvBytes, yLength, uvLength);
        vBuffer.get(yuvBytes, yLength + uvLength, uvLength);
        int width = image.getWidth();
        int height = image.getHeight();
        jniHandler.faceEyeDetection(yuvBytes, width, height, faces, eyes, eyeRegion);
//        return  res;
    }

    public static void doFaceTracking(Image image, double[] faces, double[] eyes, float[] eyeRegion){
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
        int yLength = yBuffer.capacity();
        int uvLength = uBuffer.capacity();
        byte[] yuvBytes = new byte[yLength+2*uvLength];
        yBuffer.get(yuvBytes, 0, yLength);
        uBuffer.get(yuvBytes, yLength, uvLength);
        vBuffer.get(yuvBytes, yLength + uvLength, uvLength);
        int width = image.getWidth();
        int height = image.getHeight();
        jniHandler.faceTracking(yuvBytes, width, height, faces, eyes, eyeRegion);
//        return res;
    }

    public static void fromMatToByteArray(long matAddr, byte[] returnBytes){
        jniHandler.fromMatToByteArray(matAddr, returnBytes);
    }

    public static byte[] encodeIntoJpegBytes(Image image){
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
        int yLength = yBuffer.capacity();
        int uvLength = uBuffer.capacity();
        byte[] yuvBytes = new byte[yLength+2*uvLength];
        yBuffer.get(yuvBytes, 0, yLength);
        uBuffer.get(yuvBytes, yLength, uvLength);
        vBuffer.get(yuvBytes, yLength + uvLength, uvLength);
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] encoded = jniHandler.encodeIntoJpegArray(yuvBytes, width, height, new byte[100]);
        return encoded;
    }

    public static byte[] fromMatToJpegByte(Mat image){
        MatOfByte mob=new MatOfByte();
        Imgcodecs.imencode(".jpg", image, mob);
        return mob.toArray();
    }

    public static byte[] fromMatToJpegByte2(Mat image){
        byte[] bytes = new byte[(int)(image.total() * image.channels())];
        image.get(0,0,bytes);
        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out);
        return out.toByteArray();
    }

    public static byte[] YUVtoJPEGByte(Image image, Context context) {
        byte[] nv21Bytes = YUV_420_888_to_NV21(image);
        RenderScript rs = RenderScript.create(context);
        ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
        Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21Bytes.length);
        Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);
        Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(image.getWidth()).setY(image.getHeight());
        Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

//        final Bitmap bmpout = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);

        in.copyFromUnchecked(nv21Bytes);
        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);
        Bitmap bmpout = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
        out.copyTo(bmpout);
        FileOutputStream outFile = null;
        try {
            String base = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
            String imagePath = "/"+ base + "/Download/bitmap.jpg";
            outFile = new FileOutputStream(imagePath);
            bmpout.compress(Bitmap.CompressFormat.JPEG, 100, outFile); // bmp is your Bitmap instance
            // PNG is a lossless format, the compression factor (100) is ignored
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (out != null) {
                    outFile.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        byte[] jpgBytes = new byte[out.getBytesSize()];
        out.copyTo(jpgBytes);
        return jpgBytes;
    }

    public static byte[] YUV_420_888_to_NV21(Image image) {
        ByteBuffer yBuffer = image. getPlanes()[0]. getBuffer();
        ByteBuffer uBuffer = image. getPlanes()[1]. getBuffer();
        ByteBuffer vBuffer = image. getPlanes()[2]. getBuffer();
        int ySize = yBuffer. remaining();
        int uSize = uBuffer. remaining();
        int vSize = vBuffer. remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        byte[] uBytes = new byte[uSize];
        byte[] vBytes = new byte[vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        uBuffer.get(uBytes, 0, uSize);
        vBuffer.get(vBytes, 0, vSize);
        for(int i=0; i<uSize; i++){
            nv21[ySize + 2*i] = vBytes[i];
            nv21[ySize + 2*i + 1] = uBytes[i];
        }
//        yBuffer.get(nv21, 0, ySize);
//        for(int i=0; i<vSize; i++){
//            vBuffer.get(nv21, ySize + 2*i, 1);
//            uBuffer.get(nv21, ySize + 2*i + 1, 1);
//        }
        return nv21;
    }






    static public class FaceLandmark {
        private int[] boundingBox;
        private double[] landmark;

        public FaceLandmark(){
            this.boundingBox = new int[4];
            this.landmark = new double[98];
        }

        public int[] getBoundingBox() {
            return boundingBox;
        }

        public void setBoundingBox(int[] boundingBox) {
            this.boundingBox = boundingBox;
        }

        public double[] getLandmark() {
            return landmark;
        }

        public void setLandmark(double[] landmark) {
            this.landmark = landmark;
        }
    }



}
