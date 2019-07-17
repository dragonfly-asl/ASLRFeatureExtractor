# ASLRFeatureExtractor
Code for MT Summit 2019

# OpenPose
We used an docker image based on the following dockerfile:
https://gist.github.com/moiseevigor/11c02c694fc0c22fccd59521793aeaa6

Each ASL video was processed by the following docker command:
```
nvidia-docker run \
    -v ${LOCAL_DIR}:/workspace \
    --rm -it openpose1.4:Dockerfile \
    bash -c "\
        CUDA_VISIBLE_DEVICES=0 \
        /opt/openpose-master/build/examples/openpose/openpose.bin \
        -model_folder /opt/openpose-master/models/ \
        -model_pose BODY_25 \
        -face true \
        -face_render 1 \
        -hand true \
        -hand_render 1 \
        -hand_tracking true \
        -display 0 \
        -number_people_max 1 \
        -video /workspace/${INPUT_VIDEO} \
        -write_images /workspace/${OUTPUT_DIR} \
        -write_json /workspace/${OUTPUT_DIR} \
        -write_video /workspace/${OUTPUT_DIR}/output.avi"
```

where ```LOCAL_DIR``` is the host directory containing a video file ```INPUT_VIDEO```,
and ```OUTPUT_DIR``` is where the OpenPose analysis output is created.

# DeepHand
DeepHand model and the code to evaluate the model is taken from https://github.com/neccam/TF-DeepHand.


# Feature Extraction

```
./generateFrames.sh ${LOCAL_DIR}/${INPUT_VIDEO} ${OUTPUT_DIR}
```


```
python GenerateASLRFeatures.py -i ${OUTPUT_DIR}
```

Features are stored as a matlab file with name ```${OUTPUT_DIR}.mat```





