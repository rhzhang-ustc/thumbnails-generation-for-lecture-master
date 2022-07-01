import os.path
import time
import cv2
import decord
from decord import VideoReader, VideoLoader, cpu
import operator
import numpy as np
import matplotlib.pyplot as plt

import re
from PIL import ImageStat, Image
import math
from scipy.signal import argrelextrema

from inference import get_prediction, transforms
from custom_nnmodules import *  # noqa: F401,F403
from slide_classifier_pytorch import SlideClassifier

def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal

    example:
    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """
    # print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    # print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def brightness(img):
   stat = ImageStat.Stat(img)
   r,g,b = stat.rms
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


class Frame:
    """class to hold information about each frame

    """

    def __init__(self, id, diff, brightness=0, sharpness=0, data=None):
        self.id = id
        self.diff = diff
        self.brightness = brightness
        self.sharpness = sharpness
        self.data = data

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)



def rel_change(a, b):
    x = (b - a) / max(a, b)
    print(x)
    return x


def split_key_frames_decord(path, DOWNSAMPLE=False, method="USE_LOCAL_MAXIMA"):

    USE_THRESH = False
    if method == "USE_THRESH":
        USE_THRESH = True
        # fixed threshold value
        THRESH = 0.6
    # Setting fixed threshold criteria

    USE_TOP_ORDER = False
    if method == "USE_TOP_ORDER":
        USE_TOP_ORDER = True

    USE_LOCAL_MAXIMA = False
    # Setting local maxima criteria
    if method == "USE_LOCAL_MAXIMA":
        USE_LOCAL_MAXIMA = True

    USE_MIXED_VALUE = False
    if method == "USE_MIXED_VALUE":
        USE_MIXED_VALUE = True
        coef_bright = 1  # 调节相对亮度和锐度的重要性
        coef_sharp = 1


    # Number of top sorted frames
    NUM_TOP_FRAMES = 50

    # smoothing window size
    len_window = int(20)  # 与最后的帧数成反比

    print("target video :" + path)

    frames_read = VideoReader(path, ctx=cpu(0))

    # load video and compute diff between frames
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frame_bright = []
    frame_sharp = []
    frames = []
    total_frames = len(frames_read)

    key_frames = []
    shape = frames_read[0].shape

    for i in range(total_frames):
        frame_read = frames_read[i]
        luv = cv2.cvtColor(frame_read.asnumpy(), cv2.COLOR_BGR2LUV)
        curr_frame = luv

        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)

            if USE_MIXED_VALUE:
                frame_bright.append(curr_frame[0].mean())
                sharpness = 0
                frame_sharp.append(sharpness)
                frame = Frame(i, diff_sum_mean, brightness, sharpness)
            else:
                frame = Frame(i, diff_sum_mean)

            frames.append(frame)

        prev_frame = curr_frame

    # compute keyframe
    keyframe_id_lst = []
    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_lst.append(keyframe.id)
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH:
                keyframe_id_lst.append(frames[i].id)

    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_lst.append(frames[i - 1].id)

    if USE_MIXED_VALUE:
        print("Using Mixed Metric")
        sm_value_array = smooth(np.sum([frame_diffs, coef_bright * frame_bright, coef_sharp * frame_sharp], axis=0),
                                len_window)
        frame_indexes = np.asarray(argrelextrema(sm_value_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_lst.append(frames[i - 1].id)

    key_frames = frames_read.get_batch(keyframe_id_lst).asnumpy()

    return key_frames, shape



def split_key_frames_cv2(path, DOWNSAMPLE=False, method="USE_LOCAL_MAXIMA"):

    USE_THRESH = False
    if method == "USE_THRESH":
        USE_THRESH = True
    # fixed threshold value
        THRESH = 0.6
    # Setting fixed threshold criteria


    USE_TOP_ORDER = False
    if method == "USE_TOP_ORDER":
        USE_TOP_ORDER = True

    USE_LOCAL_MAXIMA = False
    # Setting local maxima criteria
    if method == "USE_LOCAL_MAXIMA":
        USE_LOCAL_MAXIMA = True

    USE_MIXED_VALUE = False
    if method == "USE_MIXED_VALUE":
        USE_MIXED_VALUE = True
        coef_bright = 1     # 调节相对亮度和锐度的重要性
        coef_sharp = 1

    # Number of top sorted frames
    NUM_TOP_FRAMES = 1

    # smoothing window size
    len_window = int(20)    #  与最后的帧数成反比

    print("target video :" + path)
    # load video and compute diff between frames
    cap = cv2.VideoCapture(str(path))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frame_bright = []
    frame_sharp = []

    frames = []
    key_frames = []
    success, frame = cap.read()

    shape = frame.shape

    i = 0
    while (success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame_bright.append(curr_frame[0].mean())

            sharpness = 0
            frame_sharp.append(sharpness)

            frame = Frame(i, diff_sum_mean, brightness, sharpness)
            frames.append(frame)

        prev_frame = curr_frame
        i = i + 1
        success, frame = cap.read()
    cap.release()

    # compute keyframe
    keyframe_id_set = set()
    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH:
                keyframe_id_set.add(frames[i].id)
    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)

        plt.figure(figsize=(40, 20))
        plt.locator_params()
        plt.stem(sm_diff_array)
        # plt.savefig(dir + '/plot.png')

    if USE_MIXED_VALUE:
        print("Using Mixed Metric")
        sm_value_array = smooth(np.sum([frame_diffs, coef_bright * frame_bright, coef_sharp * frame_sharp], axis=0),
                                len_window)

        frame_indexes = np.asarray(argrelextrema(sm_value_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)


    # save all keyframes as image
    cap = cv2.VideoCapture(str(path))
    curr_frame = None
    keyframes = []
    success, frame = cap.read()
    idx = 0
    while (success):
        count = 0
        if idx in keyframe_id_set:
            if DOWNSAMPLE:
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
            key_frames.append(frame)
            keyframe_id_set.remove(idx)
            count += 1

        idx = idx + 1
        success, frame = cap.read()
    cap.release()
    # print(shape)  # 720*1280*3
    return key_frames, shape


def split_key_frames(path, DOWNSAMPLE=False, method="OneFramePerChunk"):
    if method == "OneFramePerChunk":
        return split_key_frames_OneFramePerChunk(path, DOWNSAMPLE)
    elif method == "cv2":
        return split_key_frames_cv2(path, DOWNSAMPLE)
    elif method == "decord":
        return split_key_frames_decord(path, DOWNSAMPLE)


def split_key_frames_OneFramePerChunk(path, DOWNSAMPLE=False, fps=60, chunk_time=5):
    if not os.path.isdir(path):
        return split_key_frames_decord(path, DOWNSAMPLE)

    model = SlideClassifier.load_from_checkpoint("ckpt/epoch=8.ckpt")
    model.eval()
    # print(model)
    # print(get_prediction(model, Image.open("test.png").convert('RGB'), None, extract_features=True))

    keyframes = []
    KeyFrames = []
    video_list = [os.path.join(path, item) for item in os.listdir(path)]
    video_list.sort(key=lambda x: int(re.split("[./]", x)[-2]))
    video_num = len(video_list)

    if video_num < 250:
        for file in video_list:
            frames_read = VideoReader(file, ctx=cpu(0))
            total_frames = len(frames_read)
            frames = []
            prev_frame = None
            step=30

            for i in range(total_frames//step):
                id=step*i
                curr_frame = frames_read[id].asnumpy()
                if curr_frame is not None and prev_frame is not None:
                    diff_sum = np.sum(cv2.absdiff(curr_frame, prev_frame))
                    frames.append(Frame(id, diff_sum))
                prev_frame = curr_frame

            frames.sort(key=operator.attrgetter("diff"), reverse=True)
            keyframes.append(frames_read[frames[0].id].asnumpy())

        return np.array(keyframes), keyframes[0].shape

    else:
        for file_index, file in enumerate(video_list):

            frames_read = VideoReader(file, ctx=cpu(0))
            total_frames = len(frames_read)
            frames = []
            prev_frame = None
            step = 30

            for i in range(total_frames // step):
                id=np.random.randint(step*i, step*(i+1)-1)
                curr_frame = frames_read[id].asnumpy()
                if curr_frame is not None and prev_frame is not None:
                    diff_sum = np.sum(cv2.absdiff(curr_frame, prev_frame))
                    frames.append(Frame(id, diff_sum))
                prev_frame = curr_frame

            frames.sort(key=operator.attrgetter("diff"), reverse=True)
            top_id = frames[0].id
            KeyFrames.append(Frame(top_id, frames[0].diff, 0, 0, frames_read[top_id]))

        KeyFrames.sort(key=operator.attrgetter("diff"), reverse=True)

        len_frame_select = min(800, len(KeyFrames))
        # print(get_prediction(model, Image.open("test.png").convert("RGB"), None, extract_features=True))

        keyframes = [KeyFrames[i].data.asnumpy() for i in range(len_frame_select)
                     if get_prediction(model, Image.fromarray(KeyFrames[i].data.asnumpy()), None, extract_features=True) != "other"]
        return np.array(keyframes), keyframes[0].shape


