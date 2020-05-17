import os
import cv2
import math
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from OpticalFlow import OpticalFlowCalculator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default= 6, help="the number of k-means")
parser.add_argument('-thresh', type=float, default= 0.001, help="the number of k-means")
parser.add_argument('-src', type=str,  default='avi/', help="the folder of input videos")
parser.add_argument('-des', type=str, default='img/', help="the folder of output frames")
args = parser.parse_args()

k_num = args.k
src_dir = args.src
des_dir = args.des
thresh = args.thresh

allavi = os.listdir(src_dir)
for i in allavi:
    src_path = os.path.join(src_dir, i)
    des_path = os.path.join(des_dir, i)
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    videoFile = cv2.VideoCapture(src_path)

    width = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
    flow = OpticalFlowCalculator(width, height, scaledown=1, move_step=16)

    ret, frame = videoFile.read()
    data = []
    plt.cla()
    count = 0
    vals = []
    frames = []
    while ret:
        xvel, yvel = flow.processFrame(frame)
        val = xvel * xvel + yvel * yvel
        if val < thresh:
            vals.append((val,count))
        frames.append(frame)
        data.append(math.log(xvel * xvel + yvel * yvel + 1))
        ret, frame = videoFile.read()
        count += 1

    x1 = range(len(data))
    plt.plot(x1,
             data,
             label='OpticalFlow',
             linewidth=3,
             color='pink',
             marker='o',
             markerfacecolor='blue',
             markersize=5)
    name, _ = osp.splitext(i)

    plt.savefig(os.path.join(des_dir, i, 'OpticalFlow.png'))

    kmeans = KMeans(n_clusters=k_num)
    kmeans.fit(vals)
    kmeans_y_predict = kmeans.predict(vals)

    indexs = [[] for k in range(k_num) ]
    final = []
    for k in range(k_num):
        min_vals = []
        min_coun = []
        for index, num in enumerate(kmeans_y_predict.tolist()):
            if num == k:
                indexs[k].append(index)
        for index in indexs[k]:
            va = vals[index][0]
            cou = vals[index][1]
            min_vals.append(va)
            min_coun.append(cou)
        min_index = min_coun[min_vals.index(min(min_vals))]
        final.append((frames[min_index], min_index))

    for fin, fi_index in final:
        fin_path = os.path.join(des_dir, i, str(fi_index)+'.jpg')
        cv2.imwrite(fin_path, fin)
    print('save to {}'.format(os.path.join(des_dir, i)))