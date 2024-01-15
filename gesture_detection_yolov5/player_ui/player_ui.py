from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QMainWindow
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
import sys
from PySide2.QtCore import Qt, QDir, QTimer
from PySide2.QtGui import QPixmap, QImage
import PySide2.QtCore
import PySide2.QtWidgets

import cv2
import os

import time
import colorsys
import numpy as np
from PIL import ImageDraw, ImageFont, Image

from yolo_ui import YOLO
import img_process


class_names = ['A', 'number 7', 'D', 'I', 'L', 'V', 'W', 'Y', 'I love you', 'number 5']
num_classes = len(class_names)
yolo = YOLO()
# from UI_QLabel import Ui_mainWindow

class Player:

    def __init__(self):
        # 从文件中加载UI定义
        # qfile_stats = QFile("UI/stats.ui")
        # qfile_stats.open(QFile.ReadOnly)
        # qfile_stats.close()

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit

        # self.ui = QUiLoader().load(qfile_stats)
        self.ui = QUiLoader().load('E:/1_TF/vscode/python/PySide(python_3.10)/UI/Player_Final.ui')  # 返回的就是QWidget的窗口对象
        # self.ui = QUiLoader().load('E:/1_TF/vscode/python/PySide(python_3.10)/UI/Player.ui')
        # vscode 里要改成绝对路径

        # self.ui.button.clicked.connect(self.handleCalc)
        # 这里的 button 和你界面设计的名字对应

    def window_init(self):
        # label 风格设置
        # 原颜色(177, 177, 177)
        self.ui.label.setStyleSheet('''background: rgba(177, 177, 177, 0.8);
                               font-family: YouYuan;
                               font-size: 18pt;
                               color: red;
                               ''')
        # self.ui.label.setStyleSheet('''font-family: YouYuan;
        #                                font-size: 18pt;
        #                                color: red;
        #                                ''')
        self.ui.label.setAlignment(Qt.AlignCenter)  # 设置字体居中现实
        self.ui.label.setText("Hello 徐彪")  # 默认显示文字
        self.ui.label2.setPixmap("../img/neu.png")  ##输入为图片路径，比如当前文件内的logo.png图片
        self.ui.label2.setFixedSize(150, 150)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        self.ui.label2.setScaledContents(True)  # 让图片自适应 label 大小

        self.ui.pushButton_1.clicked.connect(self.save_img)  # 保存图片
        self.ui.pushButton_2.clicked.connect(self.open_img)  # 打开图片

        self.ui.pushButton_10.clicked.connect(self.threshold)
        self.ui.pushButton_11.clicked.connect(self.scharrxy)
        self.ui.pushButton_12.clicked.connect(self.edge)
        self.ui.pushButton_13.clicked.connect(self.dft_high)


        # ------------------------------- Player.ui ---------------------------------
        # self.ui.pushButton.clicked.connect(self.btnLogin_clicked)
        # self.ui.pushButton_2.clicked.connect(self.OpenFileName_clicked) # 打开图片
        # self.ui.pushButton_3.clicked.connect(self.OpenFileNames_clicked) # 返回多个文件路径
        # self.pushButton_4.clicked.connect(VdoConfig)

        self.yolo_detect = 0
        self.process = 0
        self.ui.pushButton_14.clicked.connect(self.mode_yolo_detect)
        self.ui.pushButton_15.clicked.connect(self.mode_process)
        self.ui.pushButton_16.clicked.connect(self.mode_normal)

        self.cap = cv2.VideoCapture()
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        # h(色调）：x/len(self.class_names)  s(饱和度）：1.0  v(明亮）：1.0
        self.hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), self.hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.camera_button_init()
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_picture)

    def camera_button_init(self):
        self.ui.pushButton_8.clicked.connect(self.open_camera)
        self.ui.pushButton_9.clicked.connect(self.close_camera)
        self.ui.pushButton_8.setEnabled(True)
        self.ui.pushButton_9.setEnabled(False)
        self.ui.pushButton_1.setEnabled(False)

    def mode_yolo_detect(self):
        self.yolo_detect = 1
        self.process = 0

    def mode_process(self):
        self.yolo_detect = 0
        self.process = 1

    def mode_normal(self):
        self.yolo_detect = 0
        self.process = 0

    def threshold(self):
        img_gray = cv2.imread(self.image_file, cv2.IMREAD_GRAYSCALE)
        ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
        saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/img/process.jpg"
        cv2.imwrite(saveFile, thresh4)
        self.ui.label.setPixmap(saveFile)
        # self.ui.label.setFixedSize(600, 400)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小

    def scharrxy(self):
        img = cv2.imread(self.image_file, cv2.IMREAD_GRAYSCALE)
        scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharrx2 = cv2.convertScaleAbs(scharrx)
        scharry2 = cv2.convertScaleAbs(scharry)
        scharrxy = cv2.addWeighted(scharrx2, 0.5, scharry2, 0.5, 0)
        saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/img/process.jpg"
        cv2.imwrite(saveFile, scharrxy)
        self.ui.label.setPixmap(saveFile)
        # self.ui.label.setFixedSize(600, 400)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小

    def edge(self):
        img = cv2.imread(self.image_file, cv2.IMREAD_GRAYSCALE)
        v1 = cv2.Canny(img, 80, 150)
        saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/img/process.jpg"
        cv2.imwrite(saveFile, v1)
        self.ui.label.setPixmap(saveFile)
        # self.ui.label.setFixedSize(600, 400)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小

    def dft_high(self):
        img = cv2.imread(self.image_file, 0)
        img_float32 = np.float32(img)

        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  # 转换两个通道，结果非常小，映射出来*20

        # rows, cols = img.shape
        # crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
        #
        # # 高通滤波
        # mask = np.ones((rows, cols, 2), np.uint8)
        # mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        #
        # # IDFT
        # fshift = dft_shift * mask
        # f_ishift = np.fft.ifftshift(fshift)
        # img_back = cv2.idft(f_ishift)
        # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


        saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/img/process.jpg"
        # cv2.imwrite(saveFile, img_back)
        cv2.imwrite(saveFile, magnitude_spectrum)

        self.ui.label.setPixmap(saveFile)
        # self.ui.label.setFixedSize(600, 400)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小


    '''
    槽函数，该槽函数用于接收及处理 “打开摄像头” 按钮发来的按钮信号
    '''

    def open_camera(self):
        # number = self.comboBox.currentIndex()  # 获取当前复选框的索引值，其是依次排列的0、1、2、...
        self.cap = cv2.VideoCapture()
        flag = self.cap.open(0)  # 打开指定的摄像头，若成功打开，该函数会返回 Ture、否之则返回False
        if flag is False:
            QMessageBox.information(self, "警告", "该设备未正常工作", QMessageBox.Ok)
        else:
            self.ui.label.setEnabled(True)  # 此句可删
            self.ui.pushButton_8.setEnabled(False)
            self.ui.pushButton_9.setEnabled(True)
            self.ui.pushButton_1.setEnabled(True)

            self.timer.start()  # Qt计时器开始运行，不断的发出计时信号，不断的跳入到show_pic槽函数中，不断的显示图像

    '''
    槽函数，该槽函数用于接收及处理 “关闭摄像头” 按钮发来的按钮信号
    '''

    def close_camera(self):
        self.cap.release()  # 释放摄像头对象
        self.ui.pushButton_8.setEnabled(True)
        self.ui.pushButton_9.setEnabled(False)
        self.ui.pushButton_1.setEnabled(False)
        self.timer.stop()  # 停止计时器，不再显示图像帧
        self.ui.label.setText("已关闭摄像头")  # 清空Label，使之重回黑屏状态

    def show_picture(self):
        success, img = self.cap.read()
        self.img = img
        saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/img/test01.jpg"  # 带有中文的保存文件路径

        if self.yolo_detect:
            if img is not None:
                cv2.imwrite(saveFile, img)
            img = Image.open(saveFile)
            # crop, count 指定单张检测后是否对目标进行截取 和 计数
            r_image, top_boxes, top_label, score = yolo.detect_image(img, crop=False, count=False)
            font = ImageFont.truetype(font='model_data/simhei.ttf',
                                      size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
            # box = top_boxes[0]

            if success:
                # r_image = cv2.cvtColor(np.array(r_image)) # 用opencv显示要转BGR
                r_image = np.array(r_image)
                height, width = r_image.shape[:2]  # cur_frame=会返回图像的高、宽与颜色通道数，截前2
                '''
                QImage用于访问、转化图像格式操作图像等,其返回值是一个已经转化好格式的QImage对象。
                QImage支持格式枚举描述的几种图像格式，包括单色、8位、32位和alpha混合图像。当然也包括Opencv的 mat 类型数组枚举形式
                格式: QImage(枚举对象图像帧,宽,高,转化成的颜色格式)
                '''
                pixmap = QImage(r_image, width, height, QImage.Format_RGB888)
                '''
                QPixmap用于在屏幕上显示图像, 其返回值是一个QPixmap对象
                QPixmap.fromImage函数用于将QImage对象转化为QPixmap对象，注意QPixmap并非一个图像帧，而是Qt中用于图像展示的一个类对象实例
                其本身也是一种抽象的封装，可被Qt中其他的类对象进行图像显示操作。
                '''
                pixmap = QPixmap.fromImage(pixmap)
                # 获取是视频流和 label 窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                ratio = max(width / self.ui.label.width(), height / self.ui.label.height())
                pixmap.setDevicePixelRatio(ratio)  # 以适应比例将图像帧置入 Label 中进行播放
                # 视频流置于label中间部分播放
                self.ui.label.setAlignment(Qt.AlignCenter)
                self.ui.label.setPixmap(pixmap)

                # top, left, bottom, right = box
                # ww = bottom - top
                # hh = right - left
                # self.ui.label3.setPixmap(pixmap)
                # self.ui.label3.setFixedSize(ww, hh)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
                # self.ui.label3.setScaledContents(True)  # 让图片自适应 label 大小

        elif self.process == 1:

            if img is not None:
                cv2.imwrite(saveFile, img)
            img = Image.open(saveFile)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            if success:
                img = img_process.process(img)
                img = np.array(img)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                height, width = img.shape[:2]
                '''
                QImage用于访问、转化图像格式操作图像等,其返回值是一个已经转化好格式的QImage对象。
                QImage支持格式枚举描述的几种图像格式，包括单色、8位、32位和alpha混合图像。当然也包括Opencv的 mat 类型数组枚举形式
                格式: QImage(枚举对象图像帧,宽,高,转化成的颜色格式)
                '''
                pixmap = QImage(img, width, height, QImage.Format_RGB888)
                '''
                QPixmap用于在屏幕上显示图像, 其返回值是一个QPixmap对象
                QPixmap.fromImage函数用于将QImage对象转化为QPixmap对象，注意QPixmap并非一个图像帧，而是Qt中用于图像展示的一个类对象实例
                其本身也是一种抽象的封装，可被Qt中其他的类对象进行图像显示操作。
                '''
                pixmap = QPixmap.fromImage(pixmap)
                # 获取是视频流和 label 窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                ratio = max(width / self.ui.label.width(), height / self.ui.label.height())
                pixmap.setDevicePixelRatio(ratio)  # 以适应比例将图像帧置入 Label 中进行播放
                # 视频流置于label中间部分播放
                self.ui.label.setAlignment(Qt.AlignCenter)
                self.ui.label.setPixmap(pixmap)
                # self.ui.label.setFixedSize(360, 360)
        else:
            if success:
                if img is not None:
                    cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width = cur_frame.shape[:2]  # cur_frame=会返回图像的高、宽与颜色通道数，截前2
                    '''
                    QImage用于访问、转化图像格式操作图像等,其返回值是一个已经转化好格式的QImage对象。
                    QImage支持格式枚举描述的几种图像格式，包括单色、8位、32位和alpha混合图像。当然也包括Opencv的 mat 类型数组枚举形式
                    格式: QImage(枚举对象图像帧,宽,高,转化成的颜色格式)
                    '''
                    pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
                    # pixmap = QImage(cur_frame, width, height, QImage.Format_RGB32)
                    '''
                    QPixmap用于在屏幕上显示图像, 其返回值是一个QPixmap对象
                    QPixmap.fromImage函数用于将QImage对象转化为QPixmap对象，注意QPixmap并非一个图像帧，而是Qt中用于图像展示的一个类对象实例
                    其本身也是一种抽象的封装，可被Qt中其他的类对象进行图像显示操作。
                    '''
                    pixmap = QPixmap.fromImage(pixmap)
                    # 获取是视频流和 label 窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                    ratio = max(width / self.ui.label.width(), height / self.ui.label.height())
                    pixmap.setDevicePixelRatio(ratio)  # 以适应比例将图像帧置入 Label 中进行播放
                    # 视频流置于label中间部分播放
                    self.ui.label.setAlignment(Qt.AlignCenter)
                    self.ui.label.setPixmap(pixmap)
                    # self.ui.label.setFixedSize(360, 360)

    # 打开文件夹
    def btnLogin_clicked(self):
        FileDialog = QFileDialog(self.ui.pushButton)
        FileDirectory = FileDialog.getExistingDirectory(self.ui.pushButton, "标题")  # 选择目录，返回选中的路径
        print(FileDirectory)

    # 保存图片
    def save_img(self):
        # FileDialog = QFileDialog(self.ui.pushButton_1)
        # FileDirectory = FileDialog.getOpenFileNames(self.ui.pushButton_1, "保存图片")
        # print(str(FileDirectory[0]))
        # cv2.imwrite(str(FileDirectory[0]), self.img)

        file_path, _ = PySide2.QtWidgets.QFileDialog.getSaveFileName(self.ui.pushButton_2, "Save Chart", "", "PNG Files (*.png)")
        if file_path:
            cv2.imwrite(file_path, self.img)

    # 打开图片
    def open_img(self):
        FileDialog = QFileDialog(self.ui.pushButton_2)
        # 设置可以打开任何文件
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        Filter = "(*.jpg,*.png,*.jpeg,*.bmp,*.gif)|*.jgp;*.png;*.jpeg;*.bmp;*.gif|All files(*.*)|*.*"
        self.image_file, _ = FileDialog.getOpenFileName(self.ui.pushButton_2, 'open file', './',
                                                   'Image files (*.jpg *.gif *.png *.jpeg)')  # 选择目录，返回选中的路径 'Image files (*.jpg *.gif *.png *.jpeg)'
        # 判断是否正确打开文件
        if not self.image_file:
            QMessageBox.warning(self.ui.pushButton_2, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return

        print("读入文件成功")
        print(self.image_file)  # 'C:\\', 默认C盘打开
        # 设置标签的图片
        # self.ui.label.setPixmap(QPixmap.fromImage(image_file))  # 输入为图片路径，比如当前文件内的logo.png图片
        self.ui.label.setPixmap(self.image_file)
        # self.ui.label.setFixedSize(600, 400)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小

    # 多选文件
    def OpenFileNames_clicked(self):
        FileDialog = QFileDialog(self.ui.pushButton_3)
        FileDirectory = FileDialog.getOpenFileNames(self.ui.pushButton_3, "标题")  # 选择目录，返回选中的路径，在list中
        print(FileDirectory[0]) # ['E:/!_AI_self_Proj/Gesture_Detection_Yolov5/player_ui/camera.jpg']

        # img_path = ''.join(FileDirectory[0]) # E:/!_AI_self_Proj/Gesture_Detection_Yolov5/player_ui/camera.jpg
        # print(img_path)
        # img = Image.open(img_path)
        # img = np.array(img)
        # h, w = img.shape[:2]
        # self.ui.label.setPixmap(img_path)  ##输入为图片路径，比如当前文件内的logo.png图片
        # self.ui.label.setFixedSize(w, h)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        # self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小

    def open_video(self):
        FileDialog = QFileDialog(self.ui)
        # 设置可以打开任何文件
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        image_file, _ = FileDialog.getOpenFileName(self.ui.pushButton_4, 'open file', './', )
        if not image_file:
            QMessageBox.warning(self.ui.pushButton_4, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return
        print("读入文件成功")

        # camera = cv2.VideoCapture(0)
        #
        # while True:
        #     success, img = camera.read()
        #     if success:
        #         cv2.imshow("Video", img)

        #     k = cv2.waitKey(1)
        #     if k == ord('q'):
        #         break

        # # 释放摄像头并关闭OpenCV打开的窗口
        # camera.release()
        # cv2.destroyAllWindows()

        return image_file


# 视频控制
class VdoConfig:
    def __init__(self):
        # 按钮使能（否）
        self.ui = QUiLoader().load('E:/1_TF/vscode/python/PySide(python_3.10)/UI/Player.ui')  # 返回的就是QWidget的窗口对象

        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_6.setEnabled(False)
        self.ui.pushButton_7.setEnabled(False)
        self.ui.file = self.ui.open_video()
        if not self.ui.file:
            return
        self.ui.label.setText("正在读取请稍后...")
        # 设置时钟
        self.v_timer = QTimer()  # self.
        # 读取视频
        self.cap = cv2.VideoCapture(self.ui.file)
        if not self.cap:
            print("打开视频失败")
            return
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获得码率
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        print("FPS:".format(self.fps))

        self.ui.pushButton_5.setEnabled(True)
        self.ui.pushButton_6.setEnabled(True)
        self.ui.pushButton_7.setEnabled(True)
        self.ui.pushButton_5.setText("播放")
        self.ui.pushButton_6.setText("快退")
        self.ui.pushButton_7.setText("快进")

        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 连接按钮和对应槽函数，lambda表达式用于传参
        self.ui.pushButton_5.clicked.connect(self.go_pause)
        self.ui.pushButton_6.pressed.connect(lambda: self.last_img(True))
        self.ui.pushButton_6.clicked.connect(lambda: self.last_img(False))
        self.ui.pushButton_7.pressed.connect(lambda: self.next_img(True))
        self.ui.pushButton_7.clicked.connect(lambda: self.next_img(False))
        print("init OK")

    # 视频播放
    def show_pic(self):
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            # Mat格式图像转Qt中图像的方法
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(showImage))
            self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小

            # 状态栏显示信息
            self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_t, total_t = self.calculate_time(self.current_f, self.total_f, self.fps)
            self.ui.statusbar.showMessage("文件名：{}        {}({})".format(self.ui.file, current_t, total_t))

    def calculate_time(self, c_f, t_f, fps):
        total_seconds = int(t_f / fps)
        current_sec = int(c_f / fps)
        c_time = "{}:{}:{}".format(int(current_sec / 3600), int((current_sec % 3600) / 60), int(current_sec % 60))
        t_time = "{}:{}:{}".format(int(total_seconds / 3600), int((total_seconds % 3600) / 60), int(total_seconds % 60))
        return c_time, t_time

    def show_pic_back(self):
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置下一次帧为当前帧-2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_f - 2)
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(showImage))

            # 状态栏显示信息
            current_t, total_t = self.calculate_time(self.current_f - 1, self.total_f, self.fps)
            self.ui.statusbar.showMessage("文件名：{}        {}({})".format(self.ui.file, current_t, total_t))

    # 快退
    def last_img(self, t):
        self.ui.pushButton_5.setText("播放")
        if t:
            # 断开槽连接
            self.v_timer.timeout.disconnect(self.show_pic)
            # 连接槽连接
            self.v_timer.timeout.connect(self.show_pic_back)
            self.v_timer.start(int(1000 / self.fps) / 2)
        else:
            self.v_timer.timeout.disconnect(self.show_pic_back)
            self.v_timer.timeout.connect(self.show_pic)
            self.v_timer.start(int(1000 / self.fps))

    # 快进
    def next_img(self, t):
        self.ui.pushButton_5.setText("播放")
        if t:
            self.v_timer.start(int(1000 / self.fps) / 2)  # 快进
        else:
            self.v_timer.start(int(1000 / self.fps))

    # 暂停播放
    def go_pause(self):
        if self.ui.pushButton_5.text() == "播放":
            self.v_timer.stop()
            self.ui.pushButton_5.setText("暂停")
        elif self.ui.pushButton_5.text() == "暂停":
            self.v_timer.start(int(1000 / self.fps))
            self.ui.pushButton_5.setText("播放")


app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    app.addLibraryPath(os.path.join(os.path.dirname(PySide2.QtCore.__file__), "plugins")) # 不添加，加载jpg图片会失败，
stats = Player()
stats.window_init()
stats.ui.show()
app.exec_()