package org.pytorch.demo.vision;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

  public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
  public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

  private static final int INPUT_TENSOR_CHANNEL = 3;

  private static final int INPUT_TENSOR_WIDTH = 512;
  private static final int INPUT_TENSOR_HEIGHT = 256;
  private static final int TOP_K = 3;
  private static final int MOVING_AVG_PERIOD = 10;
  private static final String FORMAT_MS = "%dms";
  private static final String FORMAT_AVG_MS = "avg:%.0fms";

  private static final String FORMAT_FPS = "%.1fFPS";
  public static final String SCORES_FORMAT = "%.2f";
  public static final int[][] cmap = {{128,  64, 128},
                                      {124,   0,   0},
                                      {  0,  80, 255},
                                      {255, 160,   0},
                                      {255, 255,   0},
                                      {130, 110,  90},
                                      { 80, 110, 120},
                                      { 80, 200, 255},
                                      {157, 143, 123},
                                      {240, 160,  60},
                                      {  0,   0,   0},
                                      {220,  20,  60},
                                      {255,   0,   0},
                                      {  0,   0, 142},
                                      {  0,   0,  70},
                                      {  0,  60, 100},
                                      {  0,  80, 100},
                                      {  0,   0, 230},
                                      {119,  11,  32},
                                      {124,   0,   0},
                                      { 64, 164, 223},
                                      {153,  76,   0},
                                      {128,  64, 128},
                                      {  0,   0, 255}};
  static class AnalysisResult {

    private final String[] topNClassNames;
    private final float[] topNScores;
    private final long analysisDuration;
    private final long moduleForwardDuration;
    private final Bitmap processImage;
    public AnalysisResult(String[] topNClassNames, float[] topNScores,
                          long moduleForwardDuration, long analysisDuration, Bitmap processImage) {
      this.topNClassNames = topNClassNames;
      this.topNScores = topNScores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
      this.processImage = processImage;
    }
  }

  private boolean mAnalyzeImageErrorState;
  private ResultRowView[] mResultRowViews = new ResultRowView[TOP_K];
  private ImageView mBitmapOut;
  private TextView mFpsText;
  private TextView mMsText;
  private TextView mMsAvgText;
  private Module mModule;
  private String mModuleAssetName;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private long mMovingAvgSum = 0;
  private Queue<Long> mMovingAvgQueue = new LinkedList<>();
  Bitmap mBitmap = null;
  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_classification;
  }

  @Override
  protected TextureView getCameraPreviewTextureView() {
    return ((ViewStub) findViewById(R.id.image_classification_texture_view_stub))
        .inflate()
        .findViewById(R.id.image_classification_texture_view);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    final ResultRowView headerResultRowView =
        findViewById(R.id.image_classification_result_header_row);
    headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
    headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);

    mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
    mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
    mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);

    mFpsText = findViewById(R.id.image_classification_fps_text);
    mMsText = findViewById(R.id.image_classification_ms_text);
    mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);
    mBitmapOut = findViewById(R.id.image_classification_preview_result_image);
  }

  @Override
  protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
    mMovingAvgSum += result.moduleForwardDuration;
    mMovingAvgQueue.add(result.moduleForwardDuration);
    if (mMovingAvgQueue.size() > MOVING_AVG_PERIOD) {
      mMovingAvgSum -= mMovingAvgQueue.remove();
    }

    for (int i = 0; i < TOP_K; i++) {

      final ResultRowView rowView = mResultRowViews[i];
      rowView.nameTextView.setText(result.topNClassNames[i]);
      rowView.scoreTextView.setText(String.format(Locale.US, SCORES_FORMAT,
          result.topNScores[i]));
      rowView.setProgressState(false);
    }

    mBitmapOut.invalidate();
    mBitmapOut.setImageResource(0);
    mBitmapOut.setImageBitmap(result.processImage);
    Log.d(Constants.TAG, "New image updateed");


    mMsText.setText(String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration));
    if (mMsText.getVisibility() != View.VISIBLE) {
      mMsText.setVisibility(View.VISIBLE);
    }
    mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
    if (mFpsText.getVisibility() != View.VISIBLE) {
      mFpsText.setVisibility(View.VISIBLE);
    }

    if (mMovingAvgQueue.size() == MOVING_AVG_PERIOD) {
      float avgMs = (float) mMovingAvgSum / MOVING_AVG_PERIOD;
      mMsAvgText.setText(String.format(Locale.US, FORMAT_AVG_MS, avgMs));
      if (mMsAvgText.getVisibility() != View.VISIBLE) {
        mMsAvgText.setVisibility(View.VISIBLE);
      }
    }
  }

  protected String getModuleAssetName() {
    if (!TextUtils.isEmpty(mModuleAssetName)) {
      return mModuleAssetName;
    }
    final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
    mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
        ? moduleAssetNameFromIntent
        : "roadnet.pt";

    return mModuleAssetName;
  }

  @Override
  protected String getInfoViewAdditionalText() {
    return getModuleAssetName();
  }

  @Override
  @WorkerThread
  @Nullable
  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      if (mModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
            Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
        Log.d(Constants.TAG, "Module going to load from " + moduleFileAbsoluteFilePath);

        mModule = Module.load(moduleFileAbsoluteFilePath);
        Log.d(Constants.TAG, "Module loaded from " + moduleFileAbsoluteFilePath);

        mInputTensorBuffer =
            Tensor.allocateFloatBuffer(INPUT_TENSOR_CHANNEL * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
        mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, INPUT_TENSOR_CHANNEL, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});
      }

      float[] my_TORCHVISION_NORM_MEAN_RGB = new float[]{0.0f, 0.0f, 0.0f};
      float[] my_TORCHVISION_NORM_STD_RGB = new float[]{1.0f, 1.0f, 1.0f};

      final long startTime = SystemClock.elapsedRealtime();
      TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
          image.getImage(), rotationDegrees,
          INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
              my_TORCHVISION_NORM_MEAN_RGB,
              my_TORCHVISION_NORM_STD_RGB,
          mInputTensorBuffer, 0);

      mBitmap = BitmapFactory.decodeStream(getAssets().open("Pictures0023_leftImg8bit256.png"));
      mInputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                                                            my_TORCHVISION_NORM_MEAN_RGB,
                                                            my_TORCHVISION_NORM_STD_RGB);

      final float[] inpArray = mInputTensor.getDataAsFloatArray();

      Log.d(Constants.TAG, "Input data length =" + mInputTensor.shape().length);
      Log.d(Constants.TAG, "Input array length =" + inpArray.length);

      Log.d(Constants.TAG, "Input data =" + " " + inpArray[0] + " " +  inpArray[1] + " " +  inpArray[2] + " " +  inpArray[3] + " " +  inpArray[4]);
      Log.d(Constants.TAG, "Input data =" + " " + inpArray[0]*255 + " " +  inpArray[1]*255 + " " +  inpArray[2]*255 + " " +  inpArray[3]*255 + " " +  inpArray[4]*255);

      final long moduleForwardStartTime = SystemClock.elapsedRealtime();
      IValue outVal = mModule.forward(IValue.from(mInputTensor));
      final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
      Log.d(Constants.TAG, "Forward success Duration=" + moduleForwardDuration);

      final Tensor outputTensor = outVal.toTensor();
      long shape[] = outputTensor.shape();
      Log.d(Constants.TAG, "outputTensor array length =" + shape.length);

      int[][][] masked_image = new int[INPUT_TENSOR_CHANNEL][INPUT_TENSOR_HEIGHT][INPUT_TENSOR_WIDTH];
      int[][][] blended_image = new int[INPUT_TENSOR_CHANNEL][INPUT_TENSOR_HEIGHT][INPUT_TENSOR_WIDTH];

      Log.d(Constants.TAG, "cmap.length =" + cmap.length);
      Log.d(Constants.TAG, "cmap[0].length =" + cmap[0].length);
      Log.d(Constants.TAG, "cmap[1].length =" + cmap[1].length);
      Log.d(Constants.TAG, "masked_image[0].length =" + masked_image[0].length);
      Log.d(Constants.TAG, "masked_image[0][0].length =" + masked_image[0][0].length);

      final long[] masks = outputTensor.getDataAsLongArray();
      int maskStats[] = new int[cmap.length];
      for (int i = 0, len = maskStats.length; i < len; i++)
        maskStats[i] = 0;
      int totalPix = 0;
      int h = INPUT_TENSOR_HEIGHT;
      int w = INPUT_TENSOR_WIDTH;
      int[] pixels = new int[h*w];
      int[] blended_pixels = new int[h*w];

      int r=0,g=0,b=0;
      float blend= (float) 0.3;

      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          r = (int) (inpArray[i * w + j] * 255);
          g = (int) (inpArray[h + (i * w) + j] * 255);
          b = (int) (inpArray[2 * h + i * w + j] * 255);
          if (r > 255)
            r = 255;
          if (g > 255)
            g = 255;
          if (b > 255)
            b = 255;
          blended_image[0][i][j] = r;
          blended_image[1][i][j] = g;
          blended_image[2][i][j] = b;
        }
      }

      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w/2; j++) {
          int temp = blended_image[1][i][j];
          blended_image[1][i][j] = blended_image[1][i][w/2 + j];
          blended_image[1][i][w/2 + j] = temp;
          }
        }

      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          blended_pixels[w * i + j] = Color.argb(255, blended_image[0][i][j], blended_image[1][i][j], blended_image[2][i][j]);
        }
      }

      for (int label = 0; label < cmap.length; label++) {
        for (int i = 0; i < h; i++){
          for (int j = 0; j < w; j++) {
            int mask = (int)masks[w*i + j];
            if (mask == label) {
              maskStats[label]++;
              totalPix++;
              r = cmap[label][0];
              g = cmap[label][1];
              b = cmap[label][2];
              masked_image[0][i][j] = r;
              masked_image[1][i][j] = g;
              masked_image[2][i][j] = b;
              pixels[w*i + j] = Color.argb(255, r, r, b);
              r = (int) (blended_image[0][i][j] + (float)r * blend);
              g = (int) (blended_image[1][i][j] + (float)g * blend);
              b = (int) (blended_image[2][i][j] + (float)b * blend);

              blended_pixels[w*i + j] = Color.argb(255, r, r, b);
            }
//            else {
//              r = 0;
//              g = 0;
//              b = 0;
//            }
            //pixels[w*i + j] = Color.argb(255, 255, 255, 255);
//            if (b != 0)
//              Log.d(Constants.TAG, "R="+r + " G=" + g + " B=" + b + " pixels[i]="+pixels[i] +" h*i + j=" + (h*i + j) );
          }
        }
      }

//      for (int i =0; i < h; i++){
//        for (int j=0; j < w; j++)
//        pixels[i*w + j] = Color.argb(255, cmap[0][0], cmap[0][1], cmap[0][2]);
//      }


      Bitmap processImage = Bitmap.createBitmap(blended_pixels, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, Bitmap.Config.ARGB_8888);

//      if (bitmapWidth != resizeWidth || bitmapHeight != resizeHeight) {
//        processImage = Bitmap.createScaledBitmap(processImage, resizeWidth, resizeHeight, true);
//      }
//      processImage;

      String path = Environment.getExternalStorageDirectory().toString() + "/Roadnet";
      //String path = "/storage/sdcard0/Roadnet";

      File direct = new File(path);
      if(!direct.exists()) {
        direct.mkdir();
      }



      Log.d(Constants.TAG, "File saved in location :" + path);

      File file = new File(path, "roadnet_segmented.jpg");
      try {
        OutputStream fOut = new FileOutputStream(file);
        processImage.compress(Bitmap.CompressFormat.JPEG, 100, fOut);
        fOut.flush();
        fOut.close();
      } catch (Exception e) {
        e.printStackTrace();
      }

      InputStream fp = new FileInputStream(file);
      mBitmap = BitmapFactory.decodeStream(fp);
      mInputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
              my_TORCHVISION_NORM_MEAN_RGB,
              my_TORCHVISION_NORM_STD_RGB);
      fp.close();
      final float[] inpArray2 = mInputTensor.getDataAsFloatArray();

      Log.d(Constants.TAG, "Check Input data length =" + mInputTensor.shape().length);
      Log.d(Constants.TAG, "Check Input array length =" + inpArray2.length);

      Log.d(Constants.TAG, "Check Input data =" + " " + inpArray2[0] + " " +  inpArray2[1] + " " +  inpArray2[2] + " " +  inpArray2[3] + " " +  inpArray2[4]);
      Log.d(Constants.TAG, "Check Input data =" + " " + inpArray2[0]*255 + " " +  inpArray2[1]*255 + " " +  inpArray2[2]*255 + " " +  inpArray2[3]*255 + " " +  inpArray2[4]*255);




      for (int i = 0, len = maskStats.length; i < len; i++)
        Log.d(Constants.TAG, "maskStats i =" + i + ", count="+ maskStats[i]);

      final String[] topKClassNames = new String[TOP_K];
      final float[] topKScores = new float[TOP_K];

      Log.d(Constants.TAG, "outputTensor shape = [" + shape[0] + "," + shape[1] + "," + shape[2]);
      final int[] ixs = Utils.topK(maskStats, TOP_K);

      for (int i = 0; i < TOP_K; i++) {
        final int ix = ixs[i];
        float percentage = 0;
        if (ix < Constants.ROAD_DAMAGE_CLASSES.length)
        {
          topKClassNames[i] = Constants.ROAD_DAMAGE_CLASSES[ix];
          percentage = (float) ((100.0 *(float)maskStats[ix])/(float)totalPix);
          //topKScores[i] = maskStats[ix];///totalPix;
          topKScores[i] = percentage;
        } else {
          Log.e(Constants.TAG, "Top index lesser than total labels(" + Constants.ROAD_DAMAGE_CLASSES.length + "), Current index=" + ix);
        }

        Log.d(Constants.TAG, "topKClassNames[i] = " + topKClassNames[i] + ", topKScores[i] = " + topKScores[i] + ", maskStats[ix]=" + maskStats[ix] +
                ", ix=" + ix + " totalPix=" + totalPix + " percentage=" + percentage);
      }

      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult(topKClassNames, topKScores, moduleForwardDuration, analysisDuration, mBitmap);
    } catch (Exception e) {
      Log.e(Constants.TAG, "Error during image analysis", e);
      mAnalyzeImageErrorState = true;
      runOnUiThread(() -> {
        if (!isFinishing()) {
          showErrorDialog(v -> ImageClassificationActivity.this.finish());
        }
      });
      return null;
    }
  }

  @Override
  protected int getInfoViewCode() {
    return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (mModule != null) {
      mModule.destroy();
    }
  }
}
