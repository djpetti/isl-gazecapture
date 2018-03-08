#include <jni.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <sstream>
#include <math.h>



#define LOG_TAG "OpenCV"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))



using namespace cv;
using namespace std;

CascadeClassifier haar_face_cascade = CascadeClassifier("/sdcard/Download/haarcascades/haarcascade_frontalface_default.xml");
CascadeClassifier haar_eye_cascade = CascadeClassifier("/sdcard/Download/haarcascades/haarcascade_eye.xml");





template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}


void saveFloatArray(string path, float* array, int length)
{
    string content = to_string(*(array));
    for(int i=1; i<length; ++i)
    {
        content += " " + to_string(*(array+i));
    }
    // "/sdcard/Pictures/Android_Gaze_Data/2017_11_25/processedFolder/normData.dat"
    std::ofstream outfile(path, ofstream::app);
    outfile << content << endl;
    outfile.close();
}






extern "C" {


JNIEXPORT void JNICALL
Java_com_iai_mdf_JNInterface_MobileGazeJniInterface_getRGBMatImage(
                                                                JNIEnv *env, jobject instance,
                                                                jbyteArray yuvBytes_,
                                                                jint origWidth, jint origHeight,
                                                                jlong addrMat) {

    jbyte *yuvBytes = env->GetByteArrayElements(yuvBytes_, NULL);
    Mat& theMat = *(Mat*) addrMat;
    Mat yuvMat(origHeight + origHeight/2, origWidth, CV_8UC1, (unsigned char*) yuvBytes);
    cvtColor(yuvMat, theMat, CV_YUV2BGR_I420, 4);
//    flip(theMat, theMat, -1);    // 0 means upside-down in nexus 6p; -1 means flipping on both axises
//    cv::imwrite("/sdcard/Download/testAfter.jpg", theMat);
    // release resources
    env->ReleaseByteArrayElements(yuvBytes_, yuvBytes, 0);
}



JNIEXPORT void JNICALL
Java_com_iai_mdf_JNInterface_MobileGazeJniInterface_cropImage(JNIEnv *env, jobject instance,
                                                              jlong matAddr,
                                                              jintArray rect_,
                                                              jintArray resize_,
                                                              jfloatArray tfInput,
                                                              jlong cropMatAddr) {
    jint *rect = env->GetIntArrayElements(rect_, NULL);
    jint *resizeParas = env->GetIntArrayElements(resize_, NULL);
    jfloat *inputElem = env->GetFloatArrayElements(tfInput, NULL);
    Mat& imageMat = *(Mat*) matAddr;
    Mat& crop = *(Mat*) cropMatAddr;
    cv::Rect roi;
    roi.x = *(rect);
    roi.y = *(rect+1);
    roi.width = *(rect+2);
    roi.height = *(rect+3);
    // Crop the original image to the defined ROI
    crop = imageMat(roi);
    if(*(resizeParas) > 0 && *(resizeParas+1) > 0 ) {
        resize(crop, crop, Size(*(resizeParas+1), *(resizeParas)), 0, 0, CV_INTER_LINEAR);
    }
    uint8_t* pixelPtr = crop.data;
    int cn = crop.channels();
    Scalar_<uint8_t> bgrPixel;

    for(int r = 0; r < crop.rows; r++) {
        for(int c = 0; c < crop.cols; c++) {
            bgrPixel.val[0] = pixelPtr[r*crop.cols*cn + c*cn + 0]; // B
            bgrPixel.val[1] = pixelPtr[r*crop.cols*cn + c*cn + 1]; // G
            bgrPixel.val[2] = pixelPtr[r*crop.cols*cn + c*cn + 2]; // R
            // do something with BGR values...
            *(inputElem + r * crop.cols * 3 + c * 3 + 0) =  bgrPixel.val[2];
            *(inputElem + r * crop.cols * 3 + c * 3 + 1) =  bgrPixel.val[1];
            *(inputElem + r * crop.cols * 3 + c * 3 + 2) =  bgrPixel.val[0];
        }
    }
    int length = *(resizeParas)*(*(resizeParas+1))*3;
    // mean
    float mean = 0;
    for(int i=0; i < length; i++){
        mean += *(inputElem + i);
    }
    mean /= length;
    // stddev
    float dev = 0;
    for(int i=0; i < length; i++){
        dev += (*(inputElem + i) - mean) * (*(inputElem + i) - mean);
    }
    dev =  sqrt(dev/(length-1));
    //
    for(int i=0; i < length; i++){
        *(inputElem + i) = (*(inputElem + i) - mean) / dev;
    }

    env->ReleaseIntArrayElements(rect_, rect, 0);
    env->ReleaseFloatArrayElements(tfInput, inputElem, 0);
}

JNIEXPORT void JNICALL
Java_com_iai_mdf_JNInterface_MobileGazeJniInterface_cropImageAndSaveInput(JNIEnv *env,
                                                                          jobject instance,
                                                                          jlong matAddr,
                                                                          jintArray rect_,
                                                                          jintArray size_,
                                                                          jfloatArray tfInput,
                                                                          jlong cropMatAddr,
                                                                          jstring path_) {
    jint *rect = env->GetIntArrayElements(rect_, NULL);
    jint *resizeParas = env->GetIntArrayElements(size_, NULL);
    jfloat *inputElem = env->GetFloatArrayElements(tfInput, NULL);
    const char *path = env->GetStringUTFChars(path_, 0);
    Mat &imageMat = *(Mat *) matAddr;
    Mat &crop = *(Mat *) cropMatAddr;
    cv::Rect roi;
    roi.x = *(rect);
    roi.y = *(rect + 1);
    roi.width = *(rect + 2);
    roi.height = *(rect + 3);
    // Crop the original image to the defined ROI
    int height = *(resizeParas);
    int width = *(resizeParas + 1);
    int channel = *(resizeParas + 2);
    crop = imageMat(roi);
    if(*(resizeParas) > 0 && *(resizeParas+1) > 0 ) {
        resize(crop, crop, Size(width, height), 0, 0, CV_INTER_LINEAR);
    }
    uint8_t *pixelPtr = crop.data;
    int cn = crop.channels();
    Scalar_<uint8_t> bgrPixel;

    int LENGTH = width * height * channel;
    jfloatArray pythonArray = env->NewFloatArray(LENGTH);
    jfloat *pythonInputELem = env->GetFloatArrayElements(pythonArray, NULL);
    if (channel == 3) {
        for (int r = 0; r < crop.rows; r++) {
            for (int c = 0; c < crop.cols; c++) {
                bgrPixel.val[0] = pixelPtr[r * crop.cols * cn + c * cn + 0]; // B
                bgrPixel.val[1] = pixelPtr[r * crop.cols * cn + c * cn + 1]; // G
                bgrPixel.val[2] = pixelPtr[r * crop.cols * cn + c * cn + 2]; // R
                // do something with BGR values...
                *(inputElem + r * crop.cols * 3 + c * 3 + 0) = bgrPixel.val[2] * 2;
                *(inputElem + r * crop.cols * 3 + c * 3 + 1) = bgrPixel.val[1] * 2;
                *(inputElem + r * crop.cols * 3 + c * 3 + 2) = bgrPixel.val[0] * 2;
            }
        }
    }

    // mean
    float meanJava = 0;
    for (int i = 0; i < LENGTH; i++) {
        meanJava += *(inputElem + i);
    }
    meanJava /= LENGTH;
    // stddev
    float devJava = 0;
    for (int i = 0; i < LENGTH; i++) {
        devJava += (*(inputElem + i) - meanJava) * (*(inputElem + i) - meanJava);
    }
    devJava = sqrt(devJava / (LENGTH - 1));
    // normalize
    for (int i = 0; i < LENGTH; i++) {
        *(inputElem + i) = (*(inputElem + i) - meanJava) / devJava;
    }
    // save data for desktop Python
    if (channel == 3){
        for (int i = 0; i < LENGTH; i++) {
            int r = i / crop.cols / 3;
            int cRgb = i - r * crop.cols * 3;
            int c = cRgb / 3;
            int rgb = cRgb - c * 3;
            *(pythonInputELem + rgb * crop.cols * crop.rows + c * crop.rows + r) = *(inputElem + i);
        }
    }
    saveFloatArray(path, pythonInputELem, LENGTH);

    env->ReleaseIntArrayElements(rect_, rect, 0);
    env->ReleaseFloatArrayElements(pythonArray, pythonInputELem, 0);
    env->ReleaseFloatArrayElements(tfInput, inputElem, 0);
    env->ReleaseStringUTFChars(path_, path);
}


JNIEXPORT void JNICALL
Java_com_iai_mdf_JNInterface_MobileGazeJniInterface_fromMatToByteArray(JNIEnv *env, jobject instance,
                                                                       jlong imageAddr,
                                                                       jbyteArray imgBytes) {
    // TODO
    jbyte *bytes = env->GetByteArrayElements(imgBytes, NULL);
    Mat& imageMat = *(Mat*) imageAddr;
    env->ReleaseByteArrayElements(imgBytes, bytes, 0);
}



JNIEXPORT jintArray JNICALL
Java_com_iai_mdf_JNInterface_MobileGazeJniInterface_getRotatedRGBImage(
        JNIEnv *env, jobject instance, jbyteArray yBytes_, jbyteArray uBytes_, jbyteArray vBytes_,
        jint origWidth, jint origHeight) {
    int i = 0;
    int width = origWidth;
    int height = origHeight;
    int uvWidth = width / 2;
    int origX;
    int origY;
    int newI;
    int uvRow;
    int uvCol;
    jbyte *yBytes = env->GetByteArrayElements(yBytes_, NULL);
    jbyte *uBytes = env->GetByteArrayElements(uBytes_, NULL);
    jbyte *vBytes = env->GetByteArrayElements(vBytes_, NULL);
    jsize yLength = env->GetArrayLength(yBytes_);
    jsize uvLength = env->GetArrayLength(uBytes_);

    // upsampling the U and V array
//    jbyteArray newUByteArray = (*env)->NewByteArray(env, yLength);
//    jbyte *newUBytes = (*env)->GetByteArrayElements(env, newUByteArray, NULL);
//    jbyteArray newVByteArray = (*env)->NewByteArray(env, yLength);
//    jbyte *newVBytes = (*env)->GetByteArrayElements(env, newVByteArray, NULL);
//    for(i=0; i < uvLength-1; i++){
//        int uvWidth = width / 2;
//        int uvCol = i % uvWidth;
//        int uvRow = i / uvWidth;
//        newUBytes[2*i] = *(uBytes + i);
//        newUBytes[2*i+1] = (*(uBytes + i) + *(uBytes + i + 1))/2;
//        newVBytes[2*i] = *(vBytes + i);
//        newVBytes[2*i+1] = (*(vBytes + i) + *(vBytes + i + 1))/2;
//    }
//    newUBytes[yLength-2] = *(uBytes + uvLength -1);
//    newUBytes[yLength-1] = *(uBytes + uvLength -1);
//    newVBytes[yLength-2] = *(vBytes + uvLength -1);
//    newVBytes[yLength-1] = *(vBytes + uvLength -1);

//    // YUV -> RGB  + rotation     with the size of U & V planes
//    jintArray rgbIntArray = (*env)->NewIntArray(env, uvLength);
//    jint *rgbInt = (*env)->GetIntArrayElements(env, rgbIntArray, NULL);
//    for(i = 0; i < uvLength; ++i) {
//        uvCol = (i % uvWidth) * 2;
//        uvRow = (i / uvWidth) * 2;
//        jint R = (char)*(yBytes + uvRow*width + uvCol) + 1.40200 * ((char)*(vBytes+i)-128);
//        jint G = (char)*(yBytes + uvRow*width + uvCol) - 0.34414 * ((char)*(uBytes+i)-128) - 0.71414 * ((char)*(vBytes+i)-128);
//        jint B = (char)*(yBytes + uvRow*width + uvCol) + 1.77200 * ((char)*(uBytes+i)-128);
//        R = (R > 255)? 255 : (R < 0)? 0 : R;
//        G = (G > 255)? 255 : (G < 0)? 0 : G;
//        B = (B > 255)? 255 : (B < 0)? 0 : B;
//        jint RGB = 0xff000000 | (R << 16) | (G << 8) | B;
//        origX = i % (width/2);
//        origY = i / (width/2);
//        newI = (height/2) * ((width/2) - 1 - origX) + origY;
//        rgbInt[newI] = RGB;
//    }
    // YUV -> RGB  + rotation      with the size of Y plane
    jintArray rgbIntArray = env->NewIntArray(yLength);
    jint *rgbInt = env->GetIntArrayElements(rgbIntArray, NULL);
    for (i = 0; i < yLength; ++i) {
        uvCol = (i % width) / 2;
        uvRow = i / width / 2;
        jint R =
                (char) *(yBytes + i) + 1.40200 * ((char) *(vBytes + uvRow * uvWidth + uvCol) - 128);
        jint G = (char) *(yBytes + i) -
                 0.34414 * ((char) *(uBytes + uvRow * uvWidth + uvCol) - 128) -
                 0.71414 * ((char) *(vBytes + uvRow * uvWidth + uvCol) - 128);
        jint B =
                (char) *(yBytes + i) + 1.77200 * ((char) *(uBytes + uvRow * uvWidth + uvCol) - 128);
        R = (R > 255) ? 255 : (R < 0) ? 0 : R;
        G = (G > 255) ? 255 : (G < 0) ? 0 : G;
        B = (B > 255) ? 255 : (B < 0) ? 0 : B;
        jint RGB = 0xff000000 | (R << 16) | (G << 8) | B;
        origX = i % width;
        origY = i / width;
        newI = height * (width - 1 - origX) + origY;
        rgbInt[newI] = RGB;
    }

    env->ReleaseByteArrayElements(yBytes_, yBytes, 0);
    env->ReleaseByteArrayElements(uBytes_, uBytes, 0);
    env->ReleaseByteArrayElements(vBytes_, vBytes, 0);
//    (*env)->ReleaseByteArrayElements(env, newUByteArray, newUBytes, 0);
//    (*env)->ReleaseByteArrayElements(env, newVByteArray, newVBytes, 0);
    env->ReleaseIntArrayElements(rgbIntArray, rgbInt, 0);
    return rgbIntArray;
}


JNIEXPORT jbyteArray JNICALL
Java_com_iai_mdf_JNInterface_MobileGazeJniInterface_encodeIntoJpegArray(JNIEnv *env,
                                                                        jobject instance,
                                                                        jbyteArray yuvBytes_,
                                                                        jint origWidth, jint origHeight,
                                                                        jbyteArray encodedBytes_) {
    jbyte *encodedBytes = env->GetByteArrayElements(encodedBytes_, NULL);
    jbyte *yuvBytes = env->GetByteArrayElements(yuvBytes_, NULL);
    Mat theMat;
    Mat yuvMat(origHeight + origHeight/2, origWidth, CV_8UC1, (unsigned char*) yuvBytes);
    cvtColor(yuvMat, theMat, CV_YUV2BGR_I420, 4);
    cv::cvtColor(theMat, theMat, cv::COLOR_BGRA2BGR);
    vector<uchar> encoded;
    vector<int> compression_params;
    compression_params.push_back(95);
    imencode(".jpg", theMat, encoded, compression_params);
    uchar * encode = &encoded[0];
    jbyteArray res = env->NewByteArray(encoded.size());
    env->SetByteArrayRegion(res, 0, encoded.size(), (jbyte*)encode);
    env->ReleaseByteArrayElements(yuvBytes_, yuvBytes, 0);
    env->ReleaseByteArrayElements(encodedBytes_, encodedBytes, 0);
    return res;
}


}
