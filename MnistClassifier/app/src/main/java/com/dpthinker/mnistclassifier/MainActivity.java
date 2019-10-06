package com.dpthinker.mnistclassifier;

import android.content.ContentResolver;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.dpthinker.mnistclassifier.classifier.BaseClassifier;
import com.dpthinker.mnistclassifier.classifier.FloatClassifier;
import com.dpthinker.mnistclassifier.model.ModelConfigFactory;

import java.io.FileNotFoundException;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private final static String TAG = "MainActivity";
    ImageView mImgView;
    TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImgView = findViewById(R.id.mnist_img);
        mImgView.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(intent, 1);
            }
        });
        mTextView = findViewById(R.id.tv_prob);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            Uri uri = data.getData();
            ContentResolver cr = this.getContentResolver();
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(cr.openInputStream(uri));
                ImageView imageView = findViewById(R.id.mnist_img);
                imageView.setImageBitmap(bitmap);

                BaseClassifier classifier
                        = new FloatClassifier(ModelConfigFactory.SAVED_MODEL, this);

                String result = classifier.doClassify(bitmap);
                mTextView.setText(result);
            } catch (FileNotFoundException e) {
                Log.e(TAG, "Not found input image: " + uri.toString());
            } catch (IOException e) {
                Log.e(TAG, "Exception in init Classifier");
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}