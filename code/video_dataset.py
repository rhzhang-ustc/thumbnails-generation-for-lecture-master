from frame_process import split_key_frames
from feature_extraction import feature_extraction
import random
from torch.utils.data import Dataset
import cv2
import numpy as np
import sys


class VideoDataset(Dataset):
    def __init__(self, video_path, downsample=True, method="OneFramePerChunk"):
        # data: batch * 720 * 1280 * 3
        # method: cv2, decord, OneFramePerChunk
        # OneFrame... method needs path to be a dir
        self.method = method
        self.data, self.frame_shape = split_key_frames(path=video_path, DOWNSAMPLE=downsample, method=method)
        self.data = np.array(self.data)
        self.feature2index = {}
        self.index2frame = {}
        self.features = []
        self.features_len = 0
        self.len = len(self.data)
        self.cluster_num = 0
        self.points = []
        self.result = []
        self.thumbnail_features = []
        self.final_num = 0

        for i in range(self.len):
            self.index2frame[i] = self.data[i]

        print("data len", self.len)
        # print(np.array(self.data).shape)  # 144 * 360 * 640 * 3

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def extract_feature(self, method="CLIP"):
        self.features, self.feature2index = feature_extraction(keyframes=self.data, frames_num=self.len, method=method)
        self.features_len = len(self.features)

    def init_cluster(self, cluster_num):
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point()

    def cluster(self):

        result = []    # 一个大小为cluster_num的数列，其中每一个元素都是聚类的中心坐标
        for i in range(self.cluster_num):
            result.append([])

        for item in self.features:
            distance_min = sys.maxsize
            index = -1
            for i in range(self.cluster_num):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]

        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())

        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            self.result = result
            return
            # return result, self.points, #sum
        self.points = np.array(new_center)

        return self.cluster()

    def __sumdis(self, result):
        #  计算总距离和

        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum += self.__distance(result[i][j], self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        # 计算两点间距
        return np.sqrt(((p1-p2)**2).sum())

    def __pick_start_point(self):

        if self.cluster_num < 0 or self.cluster_num > self.features_len:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, self.features_len, step=1).tolist(), self.cluster_num)
        points = []
        for index in indexes:
            points.append(self.features[index].tolist())
        return np.array(points)

    def __sorted_points(self):
        distance_matrix = [[] for i in range(self.cluster_num)]

        for j in range(self.features_len):
            for i in range(self.cluster_num):
                if list(self.features[j]) in self.result[i]:
                    distance = self.__distance(self.points[i], self.features[j])
                    distance_matrix[i].append((distance, j))
                    break

        for i in range(self.cluster_num):
            distance_matrix[i].sort(key=lambda x: x[1])

        return distance_matrix

    def generate_thumbnail_features(self, final_num):
        distance_matrix = self.__sorted_points()
        self.final_num = final_num

        for i in range(len(distance_matrix)):
            for j in distance_matrix[i][:(final_num // self.cluster_num) +1]:
                self.thumbnail_features.append(self.features[j[1]])

    def write_thumbnails(self, output_path):

        video_size = (self.frame_shape[1], self.frame_shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, 30, video_size, True)

        # print(np.array(thumb_frames[0]).shape)    # 720*1280*3
        lst = [self.feature2index[str(self.thumbnail_features[i])] for i in range(len(self.thumbnail_features))]
        list(set(lst)).sort()

        for i in lst:
            frame = self.index2frame[i]
            frame = cv2.resize(frame, (self.frame_shape[1], self.frame_shape[0]), interpolation=cv2.INTER_CUBIC)  # 720*1280*3
            if self.method == "OneFramePerChunk" or self.method == "decord":
                r, g, b = cv2.split(frame)
                frame = cv2.merge((b, g, r))
            video.write(frame)

        video.release()

