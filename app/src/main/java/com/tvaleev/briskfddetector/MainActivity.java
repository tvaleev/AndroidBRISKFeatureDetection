package com.tvaleev.briskfddetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
//import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.features2d.Features2d;
//import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.BRISK;
import org.opencv.core.MatOfKeyPoint;


import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.view.SurfaceView;

import static org.opencv.features2d.Features2d.DrawMatchesFlags_DRAW_OVER_OUTIMG;

public class MainActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
    private static final String TAG              = "MainDetectionActivity";

    private Mat mGrey;
    private Mat mRgba;
    private BRISK mBr;
    private final static int REQUEST_CODE_ASK_PERMISSIONS = 1;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED)
            ActivityCompat.requestPermissions(this, new String[] {android.Manifest.permission.CAMERA}, REQUEST_CODE_ASK_PERMISSIONS);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.detection_activity_surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGrey = new Mat(height, width, CvType.CV_8U);
        mRgba = new Mat(height, width, CvType.CV_8UC4);

        // Create BRISK algorithm
        // 4 octaves as typical value by the original paper
        // 70 as detection threshold similar to example of this paper
        mBr = BRISK.create(70, 4);
    }

    public void onCameraViewStopped() {
        mGrey.release();
        mRgba.release();
    }

    public boolean onTouch(View v, MotionEvent event) {
         return false;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mGrey = inputFrame.gray();
        mRgba = inputFrame.rgba();

        MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
        Mat objectDescriptors = new Mat();

        // Call the BRISK implementation detect keypoints and
        // calculate the descriptors, based on the grayscale image.
        mBr.detectAndCompute(mGrey, new Mat(), objectKeypoints, objectDescriptors);

        // More interesting to get the circle around keypoints with with keypoint size and orientation
        // but in this case we will see grey ouput image.
        // Draw calculated keypoints into the original image
        Features2d.drawKeypoints(mGrey, objectKeypoints, mRgba, new Scalar( 255, 0, 0 ), DrawMatchesFlags_DRAW_OVER_OUTIMG);

        // Another quick solution how to draw keypoints
        // KeyPoint[] kpArr = objectKeypoints.toArray();
        // for (KeyPoint i : kpArr)
        //    Imgproc.drawMarker(mRgba, i.pt, new Scalar( 255, 0, 0 ));

        objectDescriptors.release();
        objectKeypoints.release();

        return mRgba;
    }
}
