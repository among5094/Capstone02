package com.example.lts_test3;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    private Interpreter tflite;
    TextView text;
    ImageView imgView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        text = findViewById(R.id.textView);
        imgView = findViewById(R.id.imgView);

        try {
            // TFLite 모델 로드 (assets 폴더 내 모델 이름 지정)
            tflite = new Interpreter(loadModelFile(this, "RHM.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("TFLite", "Failed to load model: " + e.getMessage());
        }

        // 사용할 이미지 선언
        int imageSrc = R.drawable.fish_3; // <- 여기에 사용하려는 이미지 지정

        if (tflite != null) {
            // 입력 이미지 로드 및 전처리
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), imageSrc);
            ByteBuffer inputBuffer = preprocessImage(bitmap, 416, 416); // YOLOv5 모델 입력 크기

            // 추론 수행
            float[] result = runInference(tflite, inputBuffer);

            // 결과 확인 및 임계치에 따라 메시지 처리
            boolean isTumblerDetected = isTumblerDetected(result, 0.5f); // 임계치 0.5로 판단
            if (isTumblerDetected) {
                // 텀블러가 감지된 경우 토스트 메시지 표시
                Toast.makeText(this, "텀블러가 감지되었습니다!", Toast.LENGTH_LONG).show();
            } else {
                // 텀블러가 감지되지 않은 경우
                Toast.makeText(this, "텀블러가 감지되지 않았습니다.", Toast.LENGTH_LONG).show();
            }

            // 디버깅용 로그 출력
            Log.d("TFLite", "Inference result: " + Arrays.toString(result));

            // 이미지 및 결과 화면에 표시
            imgView.setImageResource(imageSrc);

        } else {
            Log.e("TFLite", "Interpreter is not initialized.");
        }
    }

    // 모델 로드 함수
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // 이미지 전처리 함수
    private ByteBuffer preprocessImage(Bitmap bitmap, int width, int height) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * width * height * 3); // float는 4 bytes
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[width * height];
        resizedBitmap.getPixels(intValues, 0, width, 0, 0, width, height);
        int pixel = 0;
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    // 추론 함수
    private float[] runInference(Interpreter tflite, ByteBuffer inputBuffer) {
        // YOLOv5s 출력 크기 (모델 출력에 따라 조정 필요)
        float[][][] output = new float[1][10647][6]; // 출력 배열 크기 수정
        // float[1][10647][11]에서 float[1][10647][6]으로 수정
        /* 이 문제는 runInference 함수에서 모델 출력 크기를 잘못 설정했기 때문에 발생합니다.
        TensorFlow Lite 모델이 반환하는 텐서의 크기가 [1, 10647, 6]인데, 코드에서 [1, 10647, 11]로 설정했기 때문입니다.
        이를 수정하면 문제가 해결됩니다. */

        tflite.run(inputBuffer, output);
        return flattenOutput(output);
    }

    // 출력 데이터를 1차원 배열로 변환
    private float[] flattenOutput(float[][][] output) {

        boolean detected = false; // 텀블러 감지 여부

        int size = output.length * output[0].length * output[0][0].length;
        float[] flatOutput = new float[size];
        int index = 0;
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                for (int k = 0; k < output[i][j].length; k++) {
                    flatOutput[index++] = output[i][j][k];
                }
            }
        }
        return flatOutput;
    }

    // 텀블러 감지 여부 확인 함수
    private boolean isTumblerDetected(float[] results, float threshold) {

        // boolean tumblerDetected = false; // 텀블러 감지 여부
        for (int i = 0; i < results.length / 6; i++) { // 한 객체당 6개의 값
            float objectness = results[i * 6 + 4]; // 객체 존재 확률
            // 아마 배열에 y,x,w(너비),h(높이), 객체존재확률,?? // 이렇게 총6개의 값이 있는데
            // i*6+4를 한 이유는 객체존재확률을 출력하기 위함임
            // 실제로 logcat을 보면 그럼
            // results[i * 6 + 4]가 가중치


            float classConfidence = results[i * 6 + 5] * objectness; // 클래스 신뢰도 계산
            // 이 classConfidence 텀블러일 확률
            // objectness는 객체의 존재 확률이고
            // 진짜 해당 객체 자체가 있는지 없는지의 여부이고 그 여부로 인해서 그 객체가 텀블러인지 아닌지를 확인함

            System.out.println("classConfidence =>" + classConfidence);
            if (classConfidence > threshold) {
                // 신뢰도가 임계치를 초과하는 경우 텀블러가 감지된 것으로 간주
                return true;
            }
        }
        return false; // 텀블러가 감지되지 않은 경우
    }
}
