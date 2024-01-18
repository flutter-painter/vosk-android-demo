package org.vosk.demo;

import static android.content.ContentValues.TAG;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.google.common.collect.HashBiMap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.flex.FlexDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;





public class Translator {
    private static final String MODEL_PATH = "opennmt.tflite";
    private static final String VOC_SRC = "src_vocab.txt";
    private static final String VOC_TGT = "tgt_vocab.txt";

    private Interpreter tflite;
    private  HashBiMap<String, Integer> vocabSource = HashBiMap.create();
    private  HashBiMap<String, Integer> vocabTarget = HashBiMap.create();
    //private  HashBiMap<String, Integer> vocab = HashBiMap.create();
    private final Context context;

    public Translator(Context context) {
        this.context = context;
    }

    public void load() {
        loadModels();
    }

    private static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath)
            throws IOException {
        try (AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }



    private synchronized  void loadModels()  {
        try {
            AssetManager assetManager =
                    this.context.getResources().getAssets();
            ByteBuffer buffer = loadModelFile(assetManager, MODEL_PATH);
            if (buffer == null) {
                Log.e("nmt-tf", "Could not load model");
                return;
            }
            Interpreter.Options opt = new Interpreter.Options();
            FlexDelegate flexDelegate = new FlexDelegate();
            opt.setNumThreads(1);
            opt.addDelegate(flexDelegate); // https://github.com/tensorflow/flutter-tflite/issues/41 not following army tuto here
            tflite = new Interpreter(buffer, opt);
            Log.v(TAG, "TFLite model loaded.");

            // load vocab
            InputStream bufferVocabSrc = assetManager.open( VOC_SRC);
            if (bufferVocabSrc == null) {
                Log.e("nmt-tf", "Could not load source vocab");
                return;
            }
            vocabSource = readVocabSource(bufferVocabSrc);

            InputStream bufferVocabTgt = assetManager.open( VOC_TGT);
            if (bufferVocabTgt == null) {
                Log.e("nmt-tf", "Could not load target vocab");
                return;
            }
            vocabTarget = readVocabTarget(bufferVocabTgt);

        } catch (IOException ex) {
            Log.e(TAG, "Error loading TF Lite model.\n", ex);
        }
    }


    public HashBiMap<String, Integer> readVocabSource(InputStream file){
        try {
            BufferedReader brFile = new BufferedReader(new
                    InputStreamReader(file));
            String wordRead = brFile.readLine();
            int index = 0;
            while(wordRead != null){
                vocabSource.put(wordRead, index);
                index += 1;
                wordRead = brFile.readLine();
            }
            return vocabSource;
        }
        catch(IOException e) {
            return null;
        }
    }


    public HashBiMap<String, Integer> readVocabTarget(InputStream file){
        try {
            BufferedReader brFile = new BufferedReader(new
                    InputStreamReader(file));
            String wordRead = brFile.readLine();
            int index = 0;
            while(wordRead != null){
                vocabTarget.put(wordRead, index);
                index += 1;
                wordRead = brFile.readLine();
            }
            return vocabTarget;
        }
        catch(IOException e) {
            return null;
        }
    }

    //, HashBiMap<String, Integer>            vocab
    private int[] textToIds(String text){
        String[] words = text.split(" ");
        System.out.println("words.length");
        System.out.println(words.length);
        ArrayList<Integer> idsList = new ArrayList<>();
//Unknown ID is the same as vocabulary size
        int unknownId = vocabSource.size();
        System.out.println("unknownId is ");
        System.out.println(unknownId);
        for(String word : words){
            Integer id = vocabSource.get(word);
//Use Unknown ID if ID retrieved was null

            if(id == null) {
                String wordLowC = word.toLowerCase(Locale.ROOT);
                id = vocabSource.get(word.toLowerCase(Locale.ROOT));
                if(id == null) {
                id = unknownId ;
                }
            }


            System.out.println("added id is");
            System.out.println(id);
            idsList.add(id);
        }
//Turns Integer[] to int[]
        return
                idsList.stream().filter(Objects::nonNull).mapToInt(i ->
                        i).toArray();
    }
    // HashBiMap<Integer, String> inverseVocab
    private String idsToText(int[] ids){
        StringBuilder sentence = new StringBuilder();
        System.out.println("ids.length");
        System.out.println(ids.length);
        for(int id : ids){
            String word = vocabTarget
                    .inverse().get(id);
//Word isn't in the vocabulary file
            if(word == null){
                word = "<unk>";
            }
//Don't include blank words or end sentence tokens
            if("<blank>".equals(word) || "</s>".equals(word)){
                continue;
            }
            sentence.append(word).append(" ");
        }
        return sentence.toString();
    }
    public String run(String string) {
        int[] input_ids = textToIds(string);
        int[] output_ids = new int[250];
        tflite.run(input_ids, output_ids);
        String translatedSentence = idsToText(output_ids);
        System.out.println("Translated Sentence: " +
                translatedSentence);
        return translatedSentence;
    }
}


