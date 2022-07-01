from video_dataset import VideoDataset
from time import *
import numpy as np
import matplotlib.pyplot as plt
import os

DownSample = False
k = 4
final_num = 60
method = "CLIP"
init_time = []
feature_extraction_time = []
cluster_time = []
thumbnail_generation_time = []
total_time = []
video_length = []

for file in os.listdir("data/lecture_video"):
    if file == ".DS_Store":
        continue

    if os.path.isdir("data/lecture_video/" + file):
        begin_time = time()

        data_path = 'data/lecture_video/'+ str(file)
        output_path = 'data/lecture_thumbnails/' + str.split(file, '.')[0] + '_thumbnails.mp4'

        video_dataset = VideoDataset(data_path, downsample=DownSample, method="OneFramePerChunk")
        time1 = time()
        print("init time", time1-begin_time)

        video_dataset.extract_feature(method=method)
        time2 = time()
        print("feature extraction time", time2 - time1)

        video_dataset.init_cluster(k)
        video_dataset.cluster()
        time3 = time()
        print("cluster time", time3-time2)

        video_dataset.generate_thumbnail_features(final_num)
        video_dataset.write_thumbnails(output_path)

        end_time = time()
        print("thumbnail written time", end_time-time3)

