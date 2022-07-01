import os
from decord import VideoReader, cpu
import cv2
import numpy as np
# 将视频保存成5s一个的chunk，视频解码的速度为60fps

suffix = ['mp4', 'avi']

def split_video(path='data/video/', chunk_time=5, fps=30):
    # path: data/video
    for file in os.listdir(path):
        if str.split(file, ".")[-1] in suffix:
            file_name = str.split(file, ".")[0]
            file_path = os.path.join(path, str(file_name))
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            frames_read = VideoReader(os.path.join(path, file), ctx=cpu(0))

            shape = frames_read[0].shape
            frame_len = len(frames_read)
            indexes = list(range(frame_len))
            batch_size = fps * chunk_time

            for i in range(frame_len // batch_size):
                output_path = file_path + "/" + str(i) + ".mp4"
                video_size = (shape[1], shape[0])
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(output_path, fourcc, fps, video_size, True)

                try:
                    frames_batch = frames_read.get_batch(indexes[batch_size*i:batch_size*(i+1)-1]).asnumpy()
                except IndexError:
                    print("IndexError")
                    frames_batch = frames_read.get_batch(indexes[batch_size * i:-1]).asnumpy()

                for frame in frames_batch:
                    r, g, b = cv2.split(frame)
                    frame = cv2.merge((b, g, r))
                    video.write(frame)

                video.release()


if __name__ == "__main__":
    print("video preprocessing")
    data_path = "../data/lecture_video"
    split_video(data_path)