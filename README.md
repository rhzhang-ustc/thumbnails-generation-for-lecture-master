# Thumbnails-Generation-for-Lecture

Generate qualified thumbnails for lectures using pretrained CLIP.  Frames are cleaned before sending into the feature extraction model, which is inspired by [1]. An Image classification model is added [2] to filter out those uninformative frames. The model uses the method of clustering  in feature space in order to find the best moments, then generates thumbnails that can best demonstrate the key idea while maintaining the beautifulness. 

## Requirement



## Usage

Split videos first in order to use this code. The video should be stored in`../data/lecture_video` in mp4 or avi form, then run

```
python preprocess.py
```

 to get a folder which stores frames of the video. Then run

```
python main.py
```

to generate thumbnails in `../data/lecture_thumbails`

## Reference

[1] To Click or Not To Click:  Automatic Selection of Beautiful Thumbnails from Videos

[2] Automatic Generation of Lecture Notes from Slide-Based Educational Videos
