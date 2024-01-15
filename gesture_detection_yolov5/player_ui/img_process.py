import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal  # 导入sicpy的signal模块
import random as rng
import itertools
import time
import multiprocessing
from joblib import Parallel, delayed


def forfor(i, j, cr, cb, skin2):
    # global cb, skin2, cr
    # global cr
    # global cb

    # i = a[0]
    # j = a[1]
    # if (cr[i][j] >  140) and (cr[i][j] <  175) and (cb[i][j] >  100) and (cb[i][j] <  120):
    if (cr[i][j] > 134) and (cr[i][j] < 162) and (cb[i][j] > 94) and (cb[i][j] < 151):
        skin2[i][j] = 255
    else:
        skin2[i][j] = 0
    # return skin2


def process2(j, i, cr, cb, skin2):
    if (cr[i][j] > 134) and (cr[i][j] < 162) and (cb[i][j] > 94) and (cb[i][j] < 151):
        skin2[i][j] = 255
    else:
        skin2[i][j] = 0

    return skin2


def processitem(i, y, cr, cb, skin2):
    pool = multiprocessing.Pool()
    pool.map(process2, [(j, i, cr, cb, skin2) for j in range(y)])
    # if (cr[i][j] >  134) and (cr[i][j] <  162) and (cb[i][j] >  94) and (cb[i][j] <  151):
    #     skin2[i][j] =  255
    # else:
    #     skin2[i][j] =  0
    pool.close()
    pool.join()


# def aaa(a):
#     print(a)

def process(img):
    total_time = 0
    # if not cv2.cuda.getCudaEnabledDeviceCount():
    #     print("CUDA is not available. Please make sure CUDA drivers are installed.")
    #     return
    # gpu_image = cv2.cuda_GpuMat()
    # gpu_image.upload(img)

    # ----------------- 色域变换 -------------------
    # global cr, cb, skin2
    start_time = time.time()
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道分量图像

    skin2 = np.zeros(cr.shape, dtype=np.uint8)  # 根据源图像的大小创建一个全0的矩阵,用于保存图像数据
    (x, y) = cr.shape  # 获取源图像数据的长和宽

    end_time = time.time()
    total_time += end_time - start_time
    # print("色域变换", end_time - start_time)

    # 遍历图像, 判断Cr和Br通道的数值, 如果在指定范围中, 则置把新图像的点设为255,否则设为0

    # ----------------- （1）循环加速方法1，不会提速 -----------------
    # start_time = time.time()
    # for i in itertools.product(range(0, x),range(0, y)):
    #     if cr[i[0]][i[1]] > 134 and cr[i[0]][i[1]] < 162 and cb[i[0]][i[1]] > 94 and cb[i[0]][i[1]] < 151:
    #         skin2[i[0]][i[1]] =  255
    #     else:
    #         skin2[i[0]][i[1]] =  0
    # end_time = time.time()
    # # print("并行处理时间：", end_time - start_time)
    # total_time += end_time - start_time

    # ----------------- （2）循环加速方法2, multiprocessing方法（未成功） -----------------
    # a = list(itertools.product(range(0, x),range(0, y)))
    # print(a[0])
    # pool = multiprocessing.Pool()
    # results = pool.map(aaa, a)
    # pool.close()
    # pool.join()

    # a = list(itertools.product(range(0, x),range(0, y)))
    # # print(a[0])
    # pool = multiprocessing.Pool()
    # results = pool.map(for_for, a)
    # pool.close()
    # pool.join()

    # pool = multiprocessing.Pool()
    # pool.map(processitem, [(i, y, cr, cb, skin2) for i in range(0, x)])
    # pool.close()
    # pool.join()

    # ----------------- （3）循环加速方法3, joblib方法（无用） -----------------
    # start_time = time.time()
    # Parallel(n_jobs=-1)(delayed(forfor)(i,j,cr,cb,skin2) for i in range(0, x) for j in range(0, y))
    # skin2 = np.array(skin2)
    # # print(skin2.shape)
    # end_time = time.time()
    # # # print("并行处理时间：", end_time - start_time)
    # total_time += end_time - start_time

    # ----------------- 原版嵌套for -----------------
    start_time = time.time()
    for i in range(0, x):
        for j in range(0, y):
            # if (cr[i][j] >  140) and (cr[i][j] <  175) and (cb[i][j] >  100) and (cb[i][j] <  120):
            if (cr[i][j] > 134) and (cr[i][j] < 162) and (cb[i][j] > 94) and (cb[i][j] < 151):
                skin2[i][j] = 255
            else:
                skin2[i][j] = 0
    end_time = time.time()
    # print("并行处理时间：", end_time - start_time)
    total_time += end_time - start_time

    # ----------------- 提取mask -------------------
    start_time = time.time()
    skin2 = np.repeat(skin2[:, :, np.newaxis], 3, axis=2)
    skin3 = skin2 / 255
    img2 = img * skin3 / 255
    end_time = time.time()
    total_time += end_time - start_time
    # print("提取mask", end_time - start_time)

    # ----------------- 提取轮廓 -------------------
    start_time = time.time()
    ret, thresh = cv2.threshold(img2 * 255, 40, 150, cv2.THRESH_BINARY)
    cv2.imwrite("E:/!_AI_self_Proj/Gesture_Detection_Yolov5/player_ui/camera_thresh.jpg", thresh)
    thresh = cv2.imread("E:/!_AI_self_Proj/Gesture_Detection_Yolov5/player_ui/camera_thresh.jpg", cv2.IMREAD_COLOR)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    end_time = time.time()
    total_time += end_time - start_time
    # print("提取轮廓", end_time - start_time)

    # ----------------- 闭操作：尽量使轮廓平滑 -------------------
    start_time = time.time()
    draw_img2 = img2.copy()
    kernel = np.ones((5, 5), np.uint8)  # 闭运算
    draw_img2 = cv2.morphologyEx(draw_img2, cv2.MORPH_CLOSE, kernel)
    draw_img2 = cv2.morphologyEx(draw_img2, cv2.MORPH_CLOSE, kernel)
    draw_img2 = cv2.morphologyEx(draw_img2, cv2.MORPH_CLOSE, kernel)
    draw_img2 = cv2.morphologyEx(draw_img2, cv2.MORPH_CLOSE, kernel)
    draw_img2 = cv2.morphologyEx(draw_img2, cv2.MORPH_CLOSE, kernel)
    end_time = time.time()
    total_time += end_time - start_time
    # print("闭操作：尽量使轮廓平滑", end_time - start_time)

    # ----------------- 提取手部轮廓 -------------------
    start_time = time.time()
    cc = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 10000:
            cc.append(contours[i])
    # print(cc)
    # res2 = cv2.drawContours(draw_img2, cc, -1, (0, 0, 255), 2)
    # res2 = cv2.blur(res2, (3,3)) # 均值滤波，尽量平滑
    end_time = time.time()
    total_time += end_time - start_time
    # print("提取手部轮廓", end_time - start_time)

    # ----------------- 画框 -------------------
    start_time = time.time()
    if cc != []:
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        boundRect = cv2.boundingRect(cc[0])
        cv2.rectangle(img, (int(boundRect[0]), int(boundRect[1])), \
                      (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 0, 255), 2)
    end_time = time.time()
    total_time += end_time - start_time
    # print("画框", end_time - start_time)
    print("total_time: ", total_time)

    return img


def start():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        start_time = time.time()
        success, img = cap.read()
        saveFile = "E:/!_AI_self_Proj/Gesture_Detection_Yolov5/player_ui/camera.jpg"  # 带有中文的保存文件路径
        if img is None:
            break
        else:
            cv2.imwrite(saveFile, img)
        # img = cv2.imread(saveFile, cv2.IMREAD_COLOR)

        img = Image.open(saveFile)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        end_time = time.time()
        print("main_time: ", end_time - start_time)

        if success:
            process(img)
            cv2.imshow("Video", img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cap.release()  # 关闭视频
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()