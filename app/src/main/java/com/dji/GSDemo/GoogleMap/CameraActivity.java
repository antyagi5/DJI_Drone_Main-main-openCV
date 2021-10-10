package com.dji.GSDemo.GoogleMap;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.graphics.PixelFormat;
import android.graphics.Color;
import android.graphics.Canvas;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.util.AttributeSet;
import android.view.TextureView;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.TextureView.SurfaceTextureListener;
import android.view.SurfaceView;
import android.view.SurfaceHolder;
import android.graphics.Paint;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;
import android.os.AsyncTask; // AsyncTask for setting object recognition label for TextView


import com.dji.GSDemo.GoogleMap.cnn.CNNExtractorService;
import com.dji.GSDemo.GoogleMap.cnn.impl.CNNExtractorServiceImpl;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.dnn.Net;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;
import org.opencv.android.Utils;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

import dji.common.camera.SettingsDefinitions;
import dji.common.camera.SystemState;
import dji.common.error.DJIError;
import dji.common.product.Model;
import dji.common.useraccount.UserAccountState;
import dji.common.util.CommonCallbacks;
import dji.sdk.base.BaseProduct;
import dji.sdk.camera.Camera;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.useraccount.UserAccountManager;

public class CameraActivity extends org.opencv.android.CameraActivity implements SurfaceTextureListener,OnClickListener {

    private static final String TAG = MainActivity.class.getName();
    protected VideoFeeder.VideoDataListener mReceivedVideoDataListener = null;
    // Codec for video live view
    protected DJICodecManager mCodecManager = null;

    protected TextureView mVideoSurface = null;
    private Button mCaptureBtn, mShootPhotoModeBtn, mRecordVideoModeBtn, mMapBtn;
    private ToggleButton mRecordBtn;
    private TextView recordingTime;
    private Handler handler;
    BoundingBoxView mboxView;

    // OpenCV
    private static final String IMAGENET_CLASSES = "imagenet_classes.txt";
    private static final String MODEL_FILE = "pytorch_mobilenet.onnx";
    private static final String[] classNames = {"background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};

    private CameraBridgeViewBase mOpenCvCameraView;
    private Net opencvNet;
    private Net caffeNet;

    private CNNExtractorService cnnService;

    // OpenCV
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG,"OpenCV loaded successfully!");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        handler = new Handler();

        initUI();

        // The callback for receiving the raw H264 video data for camera live view
        mReceivedVideoDataListener = new VideoFeeder.VideoDataListener() {

            @Override
            public void onReceive(byte[] videoBuffer, int size) {
                if (mCodecManager != null) {
                    mCodecManager.sendDataToDecoder(videoBuffer, size);
                }
            }
        };

        // OpenCV manager initialization
        OpenCVLoader.initDebug();
        // initialize implementation of CNNExtractorService
        this.cnnService = new CNNExtractorServiceImpl();

        // Neural Network model file reading
        // obtaining converted network
        String onnxModelPath = getPath(MODEL_FILE, this);
        if (onnxModelPath.trim().isEmpty()) {
            Log.i(TAG, "Failed to get model file");
            return;
        }
        opencvNet = cnnService.getConvertedNet(onnxModelPath, TAG);
        String proto = getPath("MobileNetSSD_deploy.prototxt.txt", this);
        String weights = getPath("MobileNetSSD_deploy.caffemodel", this);
        caffeNet = Dnn.readNetFromCaffe(proto, weights);

        Camera camera = FPVApplication.getCameraInstance();

        if (camera != null) {

            camera.setSystemStateCallback(new SystemState.Callback() {
                @Override
                public void onUpdate(SystemState cameraSystemState) {
                    if (null != cameraSystemState) {

                        int recordTime = cameraSystemState.getCurrentVideoRecordingTimeInSeconds();
                        int minutes = (recordTime % 3600) / 60;
                        int seconds = recordTime % 60;

                        final String timeString = String.format("%02d:%02d", minutes, seconds);
                        final boolean isVideoRecording = cameraSystemState.isRecording();

                        CameraActivity.this.runOnUiThread(new Runnable() {

                            @Override
                            public void run() {

                                recordingTime.setText(timeString);

                                /*
                                 * Update recordingTime TextView visibility and mRecordBtn's check state
                                 */
                                if (isVideoRecording){
                                    recordingTime.setVisibility(View.VISIBLE);
                                }else
                                {
                                    recordingTime.setVisibility(View.INVISIBLE);
                                }
                            }
                        });
                    }
                }
            });

        }
    }

    protected void onProductChange() {
        initPreviewer();
        loginAccount();
    }

    private void loginAccount(){

        UserAccountManager.getInstance().logIntoDJIUserAccount(this,
                new CommonCallbacks.CompletionCallbackWith<UserAccountState>() {
                    @Override
                    public void onSuccess(final UserAccountState userAccountState) {
                        Log.e(TAG, "Login Success");
                    }
                    @Override
                    public void onFailure(DJIError error) {
                        showToast("Login Error:"
                                + error.getDescription());
                    }
                });
    }

    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // read the defined data from assets
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

    @Override
    public void onResume() {
        Log.e(TAG, "onResume");
        super.onResume();
        //initPreviewer();
        //onProductChange();

        if(mVideoSurface == null) {
            Log.e(TAG, "mVideoSurface is null");
        }
    }

    @Override
    public void onPause() {
        Log.e(TAG, "onPause");
        //uninitPreviewer();
        super.onPause();
    }

    @Override
    public void onStop() {
        Log.e(TAG, "onStop");
        super.onStop();
    }

    public void onReturn(View view){
        Log.e(TAG, "onReturn");
        this.finish();
    }

    @Override
    protected void onDestroy() {
        Log.e(TAG, "onDestroy");
        uninitPreviewer();
        super.onDestroy();
    }

    private void initUI() {
        // init mVideoSurface
        mVideoSurface = (TextureView)findViewById(R.id.video_previewer_surface);

        recordingTime = (TextView) findViewById(R.id.timer);
        mCaptureBtn = (Button) findViewById(R.id.btn_capture);
        mRecordBtn = (ToggleButton) findViewById(R.id.btn_record);
        mShootPhotoModeBtn = (Button) findViewById(R.id.btn_shoot_photo_mode);
        mRecordVideoModeBtn = (Button) findViewById(R.id.btn_record_video_mode);
        mMapBtn = (Button) findViewById(R.id.btn_map);
        mboxView = (BoundingBoxView) findViewById(R.id.boundingBoxView);


        if (null != mVideoSurface) {
            mVideoSurface.setSurfaceTextureListener(this);
        }

        mCaptureBtn.setOnClickListener(this);
        mRecordBtn.setOnClickListener(this);
        mShootPhotoModeBtn.setOnClickListener(this);
        mRecordVideoModeBtn.setOnClickListener(this);
        mMapBtn.setOnClickListener(this);
        recordingTime.setVisibility(View.INVISIBLE);



        mRecordBtn.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    startRecord();
                } else {
                    stopRecord();
                }
            }
        });
    }

    private void initPreviewer() {

        BaseProduct product = FPVApplication.getProductInstance();

        if (product == null || !product.isConnected()) {
            showToast(getString(R.string.disconnected));
        } else {
            if (null != mVideoSurface) {
                mVideoSurface.setSurfaceTextureListener(this);
            }
            if (!product.getModel().equals(Model.UNKNOWN_AIRCRAFT)) {
                VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener(mReceivedVideoDataListener);
            }
        }
    }

    private void uninitPreviewer() {
        Camera camera = FPVApplication.getCameraInstance();
        if (camera != null){
            // Reset the callback
            VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener(null);
        }
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.e(TAG, "onSurfaceTextureAvailable");
        if (mCodecManager == null) {
            mCodecManager = new DJICodecManager(this, surface, width, height);
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        Log.e(TAG, "onSurfaceTextureSizeChanged");
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        Log.e(TAG,"onSurfaceTextureDestroyed");
        if (mCodecManager != null) {
            mCodecManager.cleanSurface();
            mCodecManager = null;
        }

        return false;
    }


    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        // Convert SurfaceTexture to Bitmap
        final Bitmap frame = mVideoSurface.getBitmap();
        // Convert Bitmap to Mat
        Mat tmp = new Mat (frame.getHeight(), frame.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(frame, tmp);

        //************************************************************* //
        // Inferencing on same task as frame updating
        //************************************************************* //
        List<VisionDetRet> rectListFrame = new ArrayList<VisionDetRet>();
        rectListFrame = getDetectionRects(tmp, caffeNet);
        mboxView.setResults(rectListFrame);

        //************************************************************* //
        // Inferencing on asynchronous task as frame updating //
        // NOTE: App crashes if frames are updating too fast //
        //************************************************************* //
        //AsyncTaskRunner runner = new AsyncTaskRunner();
        //runner.execute(tmp);
    }

    // Run analysis using Neural Net and return rectangles for object detection
    public List<VisionDetRet> getDetectionRects(Mat frame, Net neuralNet) {
        final int IN_WIDTH = 300;
        final int IN_HEIGHT = 300;
        final float WH_RATIO = (float)IN_WIDTH / IN_HEIGHT;
        final double IN_SCALE_FACTOR = 0.007843;
        final double MEAN_VAL = 127.5;
        final double THRESHOLD = 0.5;
        List<VisionDetRet> rectList = new ArrayList<VisionDetRet>();

        // Get a new frame
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        // Forward image through network.
        Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR,
                new Size(IN_WIDTH, IN_HEIGHT),
                new Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/false, /*crop*/false);
        neuralNet.setInput(blob);
        Mat detections = neuralNet.forward();
        int cols = frame.cols();
        int rows = frame.rows();
        detections = detections.reshape(1, (int)detections.total() / 7);

        for (int i = 0; i < detections.rows(); ++i) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > THRESHOLD) {
                int classId = (int)detections.get(i, 1)[0];
                int left   = (int)(detections.get(i, 3)[0] * cols);
                int top    = (int)(detections.get(i, 4)[0] * rows);
                int right  = (int)(detections.get(i, 5)[0] * cols);
                int bottom = (int)(detections.get(i, 6)[0] * rows);
                String label = classNames[classId] + ": " + String.format("%.3f", confidence);
                VisionDetRet newVisionDetRect = new VisionDetRet(left, top, right, bottom, label);
                rectList.add(newVisionDetRect);
            }
        }
        return rectList;
    }

    public void showToast(final String msg) {
        runOnUiThread(new Runnable() {
            public void run() {
                Toast.makeText(CameraActivity.this, msg, Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    public void onClick(View v) {

        switch (v.getId()) {
            case R.id.btn_capture:{
                captureAction();
                break;
            }
            case R.id.btn_shoot_photo_mode:{
                switchCameraMode(SettingsDefinitions.CameraMode.SHOOT_PHOTO);
                break;
            }
            case R.id.btn_record_video_mode:{
                switchCameraMode(SettingsDefinitions.CameraMode.RECORD_VIDEO);
                break;
            }
            case R.id.btn_map:
                startActivity(new Intent(this, Waypoint1Activity.class));
            default:
                break;
        }
    }

    private void switchCameraMode(SettingsDefinitions.CameraMode cameraMode){

        Camera camera = FPVApplication.getCameraInstance();
        if (camera != null) {
            camera.setMode(cameraMode, new CommonCallbacks.CompletionCallback() {
                @Override
                public void onResult(DJIError error) {

                    if (error == null) {
                        showToast("Switch Camera Mode Succeeded");
                    } else {
                        showToast(error.getDescription());
                    }
                }
            });
        }
    }

    // Method for taking photo
    private void captureAction(){
        final Camera camera = FPVApplication.getCameraInstance();
        if (camera != null) {

            SettingsDefinitions.ShootPhotoMode photoMode = SettingsDefinitions.ShootPhotoMode.SINGLE; // Set the camera capture mode as Single mode
            camera.setShootPhotoMode(photoMode, new CommonCallbacks.CompletionCallback(){
                @Override
                public void onResult(DJIError djiError) {
                    if (null == djiError) {
                        handler.postDelayed(new Runnable() {
                            @Override
                            public void run() {
                                camera.startShootPhoto(new CommonCallbacks.CompletionCallback() {
                                    @Override
                                    public void onResult(DJIError djiError) {
                                        if (djiError == null) {
                                            showToast("take photo: success");
                                        } else {
                                            showToast(djiError.getDescription());
                                        }
                                    }
                                });
                            }
                        }, 2000);
                    }
                }
            });
        }
    }

    // Method for starting recording
    private void startRecord(){

        final Camera camera = FPVApplication.getCameraInstance();
        if (camera != null) {
            camera.startRecordVideo(new CommonCallbacks.CompletionCallback(){
                @Override
                public void onResult(DJIError djiError)
                {
                    if (djiError == null) {
                        showToast("Record video: success");
                    }else {
                        showToast(djiError.getDescription());
                    }
                }
            }); // Execute the startRecordVideo API
        }
    }

    // Method for stopping recording
    private void stopRecord(){

        Camera camera = FPVApplication.getCameraInstance();
        if (camera != null) {
            camera.stopRecordVideo(new CommonCallbacks.CompletionCallback(){

                @Override
                public void onResult(DJIError djiError)
                {
                    if(djiError == null) {
                        showToast("Stop recording: success");
                    }else {
                        showToast(djiError.getDescription());
                    }
                }
            }); // Execute the stopRecordVideo API
        }

    }

    // AsyncTask for putting object recognition label
    private class AsyncTaskRunner extends AsyncTask<Mat, Void, List<VisionDetRet>> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        protected List<VisionDetRet> doInBackground(Mat... frame) {
            List<VisionDetRet> rectListFrame = new ArrayList<VisionDetRet>();
            rectListFrame = getDetectionRects(frame[0], caffeNet);
            return rectListFrame;
        }

        protected void onPostExecute(List<VisionDetRet> rectListFrame) {
            mboxView.setResults(rectListFrame);
        }
    }

    // BoundingBoxView class for overlaying boxes
    public static class BoundingBoxView extends SurfaceView implements SurfaceHolder.Callback {

        protected SurfaceHolder mSurfaceHolder;
        private Paint mPaint;
        private boolean mIsCreated;

        public BoundingBoxView(Context context, AttributeSet attrs) {
            super(context, attrs);
            mSurfaceHolder = getHolder();
            mSurfaceHolder.addCallback(this);
            mSurfaceHolder.setFormat(PixelFormat.TRANSPARENT);
            setZOrderOnTop(true);
            mPaint = new Paint();
            mPaint.setAntiAlias(true);
            mPaint.setColor(Color.GREEN);
            mPaint.setStrokeWidth(5f);
            mPaint.setStyle(Paint.Style.STROKE);
            mPaint.setTextSize(36);
        }
        @Override
        public void surfaceChanged(SurfaceHolder surfaceHolder, int format, int width, int height) {
        }
        @Override
        public void surfaceCreated(SurfaceHolder surfaceHolder) {
            mIsCreated = true;
        }
        @Override
        public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
            mIsCreated = false;
        }
        public void setResults(List<VisionDetRet> detRects)
        {
            if (!mIsCreated) {
                return;
            }
            Canvas canvas = mSurfaceHolder.lockCanvas();
            //Clear the last frame.
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
            canvas.drawColor(Color.TRANSPARENT);
            for (VisionDetRet detRet : detRects) {
                Rect rect = new Rect(detRet.getLeft(), detRet.getTop(), detRet.getRight(), detRet.getBottom());
                canvas.drawRect(rect, mPaint);
                canvas.drawText(detRet.getLabel(), detRet.getLeft()+(detRet.getRight()/4), detRet.getBottom(), mPaint);
            }
            mSurfaceHolder.unlockCanvasAndPost(canvas);
        }
    }

    // VisionDetRet class to create list of rectangles and labels
    public final class VisionDetRet {
        private int mLeft;
        private int mTop;
        private int mRight;
        private int mBottom;
        private String mLabel;
        VisionDetRet() {}
        public VisionDetRet(int l, int t, int r, int b, String lbl) {
            mLeft = l;
            mTop = t;
            mRight = r;
            mBottom = b;
            mLabel = lbl;
        }
        public int getLeft() {
            return mLeft;
        }
        public int getTop() {
            return mTop;
        }
        public int getRight() {
            return mRight;
        }
        public int getBottom() {
            return mBottom;
        }
        public String getLabel() { return mLabel; }
    }




}