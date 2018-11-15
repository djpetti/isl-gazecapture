package com.iai.mdf.DependenceClasses;

import android.content.Context;

import com.iai.mdf.R;

import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by Mou on 3/2/2018.
 */

public class DeviceProfile {

    /**
     * positve X is from left to right along short edge;
     * positve Y is from left to right along long edge;
     */

    private String  DeviceName;
    private double  cameraOffsetX;
    private double  cameraOffsetY;
    private double  screenSizeX;
    private double  screenSizeY;
    private int     collectionCaptureDelayTime;
    private int     demoCaptureDelayTime;
    private int     imageRotation;




    public static ArrayList<DeviceProfile> loadDeviceProfileList(Context ctxt){
        try {
            // read .xml file into a string
            ArrayList<DeviceProfile> devices = new ArrayList<>();
            XmlPullParser parser = ctxt.getResources().getXml(R.xml.device_profile);
            while (parser.next() != XmlPullParser.END_DOCUMENT) {
                int eventType = parser.getEventType();
                if (eventType == XmlPullParser.START_TAG && parser.getName().equalsIgnoreCase("device")) {
                    DeviceProfile profile = new DeviceProfile();
                    while (true) {
                        parser.next();
                        if (parser.getEventType() == XmlPullParser.END_TAG && parser.getName().equalsIgnoreCase("device")) {
                            break;
                        } else if (parser.getEventType() == XmlPullParser.END_TAG) {
                            continue;
                        }
                        if (parser.getName().equalsIgnoreCase("name")) {
                            parser.next();
                            profile.setDeviceName(parser.getText());
                        } else if (parser.getName().equalsIgnoreCase("camera_location_x_in_cm")) {
                            parser.next();
                            profile.setCameraOffsetX(Double.parseDouble(parser.getText()));
                        } else if (parser.getName().equalsIgnoreCase("camera_location_y_in_cm")) {
                            parser.next();
                            profile.setCameraOffsetY(Double.parseDouble(parser.getText()));
                        } else if (parser.getName().equalsIgnoreCase("screen_size_x_in_cm")) {
                            parser.next();
                            profile.setScreenSizeX(Double.parseDouble(parser.getText()));
                        } else if (parser.getName().equalsIgnoreCase("screen_size_y_in_cm")) {
                            parser.next();
                            profile.setScreenSizeY(Double.parseDouble(parser.getText()));
                        } else if (parser.getName().equalsIgnoreCase("capture_delay")) {
                            parser.next();
                            profile.setCollectionCaptureDelayTime(Integer.parseInt(parser.getText()));
                        }
                    }
                    devices.add(profile);
                }
            }
            return devices;
        } catch (XmlPullParserException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static DeviceProfile getProfileByName(ArrayList<DeviceProfile> list, String deviceName){
        for (DeviceProfile each : list) {
            if (each.getDeviceName().equalsIgnoreCase(deviceName)){
                return each;
            }
        }
        return null;
    }




    public String getDeviceName() {
        return DeviceName;
    }

    public void setDeviceName(String deviceName) {
        DeviceName = deviceName;
    }

    public double getCameraOffsetX() {
        return cameraOffsetX;
    }

    public void setCameraOffsetX(double cameraOffsetX) {
        this.cameraOffsetX = cameraOffsetX;
    }

    public double getCameraOffsetY() {
        return cameraOffsetY;
    }

    public void setCameraOffsetY(double cameraOffsetY) {
        this.cameraOffsetY = cameraOffsetY;
    }

    public double getScreenSizeX() {
        return screenSizeX;
    }

    public void setScreenSizeX(double screenSizeX) {
        this.screenSizeX = screenSizeX;
    }

    public double getScreenSizeY() {
        return screenSizeY;
    }

    public void setScreenSizeY(double screenSizeY) {
        this.screenSizeY = screenSizeY;
    }

    public int getCollectionCaptureDelayTime() {
        return collectionCaptureDelayTime;
    }

    public void setCollectionCaptureDelayTime(int collectionCaptureDelayTime) {
        this.collectionCaptureDelayTime = collectionCaptureDelayTime;
    }
}
