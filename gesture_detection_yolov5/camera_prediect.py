#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
import colorsys
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image

from yolo_camera import YOLO

if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在yolo.py_423行左右处的YOLO_ONNX
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"

    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    class_names = ['A', 'number 7', 'D', 'I', 'L', 'V', 'W', 'Y', 'I love you', 'number 5']
    num_classes = len(class_names)

    if mode != "predict_onnx":
        yolo = YOLO()
    else:
        pass
        # yolo = YOLO_ONNX()

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        # h(色调）：x/len(self.class_names)  s(饱和度）：1.0  v(明亮）：1.0
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        while True:
            print()
            success, img = camera.read()
            saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/img/test01.jpg"  # 带有中文的保存文件路径
            # saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/dataset/hand_gesture_dataset/images/train/0141.png"  # 带有中文的保存文件路径
            # saveFile = r"E:/!_AI_self_Proj/Gesture_Detection_Yolov5/dataset/hand_gesture_dataset/images/val/1393.png"
            if img is None:
                break
            else:
                cv2.imwrite(saveFile, img)

            img = Image.open(saveFile)


            r_image, top_boxes, top_label, score = yolo.detect_image(img, crop=crop, count=count)
            # r_image = yolo.detect_image(img, crop=crop, count=count)
            # r_image.show()

            font = ImageFont.truetype(font='model_data/simhei.ttf',
                                      size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
            box = top_boxes
            # [batch, c_num, 7]，x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # for i, c in list(enumerate(top_label)):
            #     predicted_class = class_names[int(c)]
            #     box = top_boxes[i]
            #
            #     top, left, bottom, right = box
            #
            #     top = max(0, np.floor(top).astype('int32'))
            #     left = max(0, np.floor(left).astype('int32'))
            #     bottom = min(img.size[1], np.floor(bottom).astype('int32'))
            #     right = min(img.size[0], np.floor(right).astype('int32'))
            #
            #     label = '{} {:.2f}'.format(predicted_class, score)
            #     draw = ImageDraw.Draw(img)
            #     label_size = draw.textsize(label, font)
            #     label = label.encode('utf-8')
            #     print(label, top, left, bottom, right)
            #
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #
            #     for j in range(2):
            #         draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            #     draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            #     del draw

            if success:
                r_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
                cv2.imshow("Video", r_image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        # 释放摄像头并关闭OpenCV打开的窗口
        camera.release()
        cv2.destroyAllWindows()