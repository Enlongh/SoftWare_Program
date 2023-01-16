import array
import sys

import numpy as np
from PyQt5.QtWidgets import (QApplication, QListWidgetItem, QMainWindow, QFileDialog, QGraphicsItem,
                             QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsView, QInputDialog, QLCDNumber)
from PyQt5.QtGui import QPixmap, QImage, QIcon, qRed, qBlue, qGreen, qRgb
from PyQt5.QtCore import QTimer, QDateTime, QSize, QThread, pyqtSignal
from ui.main_wins import *
from ui.Set3D import *
from ImageLoad.imageview import *
import os
import cv2
from ui.sub_window import *
from data_utils.inference import *
# from data_utils.inference import *
# from data_utils.visualize import *
import time

from PIL import Image


# 文本窗体暂时未修改

class UIOperate(QMainWindow, Ui_MainWindow):
    img_path_glob = []
    def __init__(self, parent=None):
        super(UIOperate, self).__init__(parent)
        self.setupUi(self)

        self.read_file.triggered.connect(self.load_from_paths)  # openimage
        self.save_file.triggered.connect(self.saveimage)

        self.currentImgIdx = 0
        self.image_paths = []
        self.infer_image = [0] * 1000

        # 展示推理图像按钮
        # self.Button_display_oldimg.clicked.connect(self.load_infer_Image)
        self.imgList.itemSelectionChanged.connect(self.loadImage)

        # 图像处理
        self.actionGaussian.triggered.connect(self.Gaussianblur)
        self.actionBrightnessContrast.triggered.connect(self.Hist)
        self.actionGamma.triggered.connect(self.Gammatransform)
        self.actionNoneTranspose.triggered.connect(self.Recover)

        # self.view = QGraphicsView(self)
        # self.view.setGeometry(0, 0, 710, 650)
        self.graphicsScene = QGraphicsScene()  # 创建场景
        self.result_graphicsScene = QGraphicsScene()  # 创建场景
        self.imgFile = None

        self.action_edit_comments.triggered.connect(self.editcomments)
        self.commentpath = r'C:\Users\Enlong\Desktop\commnet.txt'  # 考虑可以选择保存路径

        self.action_view_Comments.triggered.connect(self.viewcomments)
        self.sub_window = Widget_Test()

        '''推理操作'''
        self.infer_2.clicked.connect(self.infer_current)  # 提取当前帧
        # self.inferNet = loadNetThread()  # 创建一个线程——还是不能在这里创建
        self.pushButton_3.clicked.connect(self.infer_all)  # 提取所有帧
        # self.inferNet = inferNet

        # 两个操作是不是有冲突
        self.pushButton_10.clicked.connect(self.exitNet)
        self.lineEdit.setText('-150')
        self.lineEdit_2.setText('350')
        self.lineEdit.returnPressed.connect(self.cut_window)
        self.lineEdit_2.returnPressed.connect(self.cut_window)
        self.time_q = QTimer(self)
        self.time_q.setInterval(100)
        self.start_time = 0

        '3D模型展示'
        # self.save_file_exi.triggered.connect(self.view3D)
        self.Button_display_oldimg.clicked.connect(self.view3D)
        # self.img3D = np.ones((512, 512))   #33333333333333333333333333333333333333记得改一下
        self.img3D = None
        '''主窗口标题'''
        self.setWindowTitle("智能肿瘤分割")
        # self.time.timeout.connect(self.refresh)

    # 感觉需要读取文件夹内容，然后点击哪一个old_img就会显示哪一个
    # def openimage(self):
    #     imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
    #     # #路径和种类
    #     # print(imgName)  #C:/Users/Enlong/Desktop/1.jpg
    #     # print(type(imgName))
    #     ##不需要转换
    #     # imgName = imgName.replace('/', '\\')  #C:\Users\Enlong\Desktop\1.jpg
    #     # print(imgName)
    #     pix = QPixmap(imgName)
    #     pixmapItem = QGraphicsPixmapItem(pix)  # 创建像素图元
    #     self.graphicsScene.addItem(pixmapItem)
    #
    #     self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  #关闭滚动条
    #     self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  #不关闭滚动条会挡住图像
    #
    #     self.old_img.setScene(self.graphicsScene)
    #     self.old_img.show()
    #     # jpg = QPixmap(imgName).scaled(self.old_img.width(), self.old_img.height())
    #     # self.old_img.setPixmap(jpg)

    def QImage2CV(self, q_img):
        tmp = q_img
        # 使用numpy创建空的图象
        cv_image = np.zeros((tmp.height(), tmp.width(), 3), dtype=np.uint8)
        for row in range(0, tmp.height()):
            for col in range(0, tmp.width()):
                r = qRed(tmp.pixel(col, row))
                g = qGreen(tmp.pixel(col, row))
                b = qBlue(tmp.pixel(col, row))
                cv_image[row, col, 0] = b
                cv_image[row, col, 1] = g
                cv_image[row, col, 2] = r
        # cv_image = cv2.merge(cv_image)
        return cv_image

    def CV2QImage(self, cv_image):
        width = cv_image.shape[1]  # 获取图片宽度
        height = cv_image.shape[0]  # 获取图片高度

        pixmap = QPixmap(width, height)  # 根据已知的高度和宽度新建一个空的QPixmap,
        qimg = pixmap.toImage()  # 将pximap转换为QImage类型的qimg

        # 循环读取cv_image的每个像素的r,g,b值，构成qRgb对象，再设置为qimg内指定位置的像素
        for row in range(0, height):
            for col in range(0, width):
                b = cv_image[row, col, 0]
                g = cv_image[row, col, 1]
                r = cv_image[row, col, 2]
                pix = qRgb(int(r), int(g), int(b))
                qimg.setPixel(col, row, pix)
        return qimg  # 转换完成，返回

    '''图像处理部分'''

    def Gaussianblur(self):  # 高斯滤波
        # 获取old图像
        rect = self.old_img.scene().sceneRect()  # 这个获取的是大小
        pixmap = QImage(rect.height(), rect.width(), QtGui.QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(pixmap)
        # painter.begin(pixmap)
        rectf = QRectF(0, 0, pixmap.rect().height(), pixmap.rect().width())
        self.old_img.scene().render(painter, rectf, rect)
        painter.end()
        # 图像格式转换
        cv_img = self.QImage2CV(pixmap)  # numpy
        # print(type(cv_img))
        img_guss = cv2.GaussianBlur(cv_img, (3, 3), 1, 2)
        # cv2.imshow("Guss", cv_img)  # 显示图片
        # cv2.waitKey(0)  # 等待按键
        q_img_after = self.CV2QImage(img_guss)
        pix_after = QPixmap(q_img_after)
        pixmapItem = QGraphicsPixmapItem(pix_after)  # 创建像素图元
        self.graphicsScene.addItem(pixmapItem)
        self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
        self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像
        self.old_img.setScene(self.graphicsScene)
        self.old_img.show()

    def Hist(self):
        # 获取old图像
        rect = self.old_img.scene().sceneRect()  # 这个获取的是大小
        pixmap = QImage(rect.height(), rect.width(), QtGui.QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(pixmap)
        # painter.begin(pixmap)
        rectf = QRectF(0, 0, pixmap.rect().height(), pixmap.rect().width())
        self.old_img.scene().render(painter, rectf, rect)
        painter.end()
        # 图像格式转换
        cv_img = self.QImage2CV(pixmap)  # numpy类型
        # hist = cv2.calcHist([cv_img], [0], None, [256], [0, 256])  #只是计算直方图的
        # cv2.threshold(cv_img, 185, 255, 0, cv_img)  #不需要等于，直接处理即可——二值化，然后显示直接使用cv_img
        # print(hist)
        # print(type(hist))
        # cv2.imshow("hist", cv_img)  # 显示图片
        # cv2.waitKey(0)  # 等待按键
        b, g, r = cv2.split(cv_img)
        b1 = cv2.equalizeHist(b)
        g1 = cv2.equalizeHist(g)
        r1 = cv2.equalizeHist(r)
        output = cv2.merge([b1, g1, r1])  # 可以了，彩色直方图均衡
        q_img_after = self.CV2QImage(output)  ###########################直接传入cv_img就可以？？？？
        pix_after = QPixmap(q_img_after)
        pixmapItem = QGraphicsPixmapItem(pix_after)  # 创建像素图元
        self.graphicsScene.addItem(pixmapItem)
        self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
        self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像
        self.old_img.setScene(self.graphicsScene)
        self.old_img.show()

    def Gammatransform(self):  # 伽马变换
        # 获取old图像
        rect = self.old_img.scene().sceneRect()  # 这个获取的是大小
        pixmap = QImage(rect.height(), rect.width(), QtGui.QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(pixmap)
        # painter.begin(pixmap)
        rectf = QRectF(0, 0, pixmap.rect().height(), pixmap.rect().width())
        self.old_img.scene().render(painter, rectf, rect)
        painter.end()

        # 图像格式转换
        cv_img = self.QImage2CV(pixmap)  # numpy
        # print(type(cv_img))
        # img_guss = cv2.GaussianBlur(cv_img, (3, 3), 1, 2)
        fI = cv_img / 255.0  # 图像归一化
        gamma = 0.4  # 伽马变化
        img_gamma = np.power(fI, gamma) * 255.0
        # cv2.imshow("Guss", img_gamma)  # 显示图片
        # cv2.waitKey(0)  # 等待按键

        q_img_after = self.CV2QImage(img_gamma)
        pix_after = QPixmap(q_img_after)
        pixmapItem = QGraphicsPixmapItem(pix_after)  # 创建像素图元
        self.graphicsScene.addItem(pixmapItem)
        # self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
        # self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像
        self.old_img.setScene(self.graphicsScene)
        self.old_img.show()

    def Recover(self):  # 复原
        img_path = self.textBrowser_2.toPlainText()
        # print(img_path)
        pix = QPixmap(img_path)
        pixmapItem = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.graphicsScene.addItem(pixmapItem)
        self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
        self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像

        self.old_img.setScene(self.graphicsScene)
        self.old_img.show()

    def cut_window(self):  # 更新窗位大小
        window_min = self.lineEdit.text()
        window_max = self.lineEdit_2.text()
        window = "[" + window_min + " " + window_max + "]"
        self.label_16.setText(window)

    def load_from_paths(self):  # 加载整个文件夹图片进入
        img_dir = QFileDialog.getExistingDirectory(self, "打开文件夹", os.getcwd())  # 最后一个cwd是当前程序位置
        if img_dir == '':
            return
        filenames = os.listdir(img_dir)
        for file in filenames:
            if file[-4:] == ".png" or file[-4:] == ".jpg":
                self.image_paths.append(os.path.join(img_dir, file))
        self.image_paths.sort(key=lambda x: int(x.split('.')[0].split("\\")[1]))  # 文件名 按数字排序
        for img_path in self.image_paths:
            if os.path.isfile(img_path):
                img_name = os.path.basename(img_path)
                # print(img_name)
                item = QListWidgetItem(QIcon(img_path), img_name)
                # item.setText(img_name)
                # item.setIcon(QIcon(img_path))
                self.imgList.addItem(item)
        self.label_20.setText(str((len(self.image_paths))))
        self.imgList.show()

    def loadImage(self):  # 点击图片显示
        self.currentImgIdx = self.imgList.currentIndex().row()
        if self.currentImgIdx in range(len(self.image_paths)):
            self.currentImg = QPixmap(self.image_paths[self.currentImgIdx]).scaledToHeight(512)
            self.textBrowser_2.append(str(self.image_paths[self.currentImgIdx]))
            pixmapItem = QGraphicsPixmapItem(self.currentImg)  # 创建像素图元
            self.graphicsScene.addItem(pixmapItem)
            self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
            self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像
            self.old_img.setScene(self.graphicsScene)
            self.old_img.show()
            self.label_10.setText(str(self.currentImgIdx + 1))
            # self.label_15.setText(str(self.currentImg.height()))

    def load_infer_Image(self):  # 加载推理图像
        self.currentImgIdx = self.imgList.currentIndex().row()
        if type(self.infer_image[self.currentImgIdx]) is np.ndarray:
            if self.currentImgIdx in range(len(self.image_paths)):
                img = self.infer_image[self.currentImgIdx]
                img = img * 127.5
                img = img.astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                self.zoomscale = 1  # 图片放缩尺度
                frame = QImage(img, x, y, x * 3, QImage.Format_RGB888)
                pix = QPixmap.fromImage(frame)
                pixmapItem = QGraphicsPixmapItem(pix)  # 创建像素图元
                self.result_graphicsScene.addItem(pixmapItem)
                self.result_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
                self.result_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像
                self.result_img.setScene(self.result_graphicsScene)
                self.result_img.show()
                organ_size = np.sum(self.infer_image[self.currentImgIdx] == 1)
                tumor_size = np.sum(self.infer_image[self.currentImgIdx] == 2)
                self.label_21.setText(str(organ_size))
                self.label_22.setText(str(tumor_size))

        # def openimage(self):
        #     imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        #     print("imgName: ", imgName)  # 文件路径——C:/Users/Enlong/OneDrive - 60knxp/学习资料/软件工程/Data/39/0.png
        #     print("imhType: ", imgType)  # 文件类型——*.png
        #     pix = QPixmap(imgName)
        #     pixmapItem = QGraphicsPixmapItem(pix)  # 创建像素图元
        #     self.graphicsScene.addItem(pixmapItem)
        #     self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
        #     self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像
        #     self.old_img.setScene(self.graphicsScene)
        #     self.old_img.show()

    # 另存为（其它格式）和保存（图片）？？
    def saveimage(self):  # 保存到选择路径
        # # rect = QGraphicsView.viewport(self.dlg.gvPointRecords).rect()
        # rect = self.old_img.viewport(self.old_img).rect()
        # pixmap = QPixmap(rect.size())
        # painter = QPainter(pixmap)
        # painter.begin(pixmap)
        # # self.dlg.gvPointRecords.render(painter, QRectF(pixmap.rect()), rect)
        # self.old_img.render(painter, QRectF(pixmap.rect()), rect)
        # painter.end()
        # if self.imgFile:
        #     imgFile, _ = QFileDialog.getSaveFileName(self.dlg, "保存图像", os.path.dirname(self.imgFile),
        #                                              "影像 (*.png *.jpg)")
        # else:
        #     imgFile, _ = QFileDialog.getSaveFileName(self.dlg, "保存图像", './', "影像 (*.png)")
        # if imgFile:
        #     img = pixmap.save(imgFile)
        # self.imgFile = imgFile

        # 将old_image改为result_img就好了
        if self.result_img.scene() is not None:
            rect = self.result_img.scene().sceneRect()  # 这个获取的是大小
            pixmap = QImage(rect.height(), rect.width(), QtGui.QImage.Format_ARGB32_Premultiplied)
            painter = QPainter(pixmap)
            painter.begin(pixmap)
            rectf = QRectF(0, 0, pixmap.rect().height(), pixmap.rect().width())
            self.result_img.scene().render(painter, rectf, rect)
            painter.end()
            # pixmap.save('C:/Users/Enlong/Desktop/file.jpg')

            if self.imgFile:
                imgFile, _ = QFileDialog.getSaveFileName(self, "保存图像", os.path.dirname(self.imgFile),
                                                         "图片类型 (*.png *.jpg)")
            else:
                imgFile, _ = QFileDialog.getSaveFileName(self, "保存图像", './', "图片类型 (*.png *.jpg)")
            if imgFile:
                img = pixmap.save(imgFile)
            self.imgFile = imgFile

    # 需要实现打开后再编辑吗(之前有)？需要区分用户吗？
    def editcomments(self):  # 编辑评论
        text, ok = QInputDialog.getMultiLineText(self, 'Comment Input Dialog', '请输入评价：')
        # commentpath = r'C:\Users\Enlong\Desktop\commnet.txt'  #加上r，声明字符串，不用转义处理
        # commentpath = 'C:\\Users\\Enlong\\Desktop\\commnet.txt'  #绝对路径的处理
        if ok and text:
            # self.GetstrlineEdit.setText(str(text))
            # print(text)
            file = open(self.commentpath, 'w')  # 是需要覆盖前面的吗，还是只是追加到后面'a'
            file.write(text)

    def viewcomments(self):
        if self.commentpath:
            file = open(self.commentpath, 'r')
            text = file.read()
            # print(text)
        self.sub_window.label.setText(text)
        self.sub_window.showwidget()

    '''推理'''

    def infer_current(self):  # 推理当前图像
        # start_time = time.time()
        # self.start_time = QDateTime.currentDateTime()
        # self.time_q.timeout.connect(self.refresh)
        # self.time_q.start()
        # 获取old图像
        if self.old_img.scene() is not None:
            rect = self.old_img.scene().sceneRect()  # 这个获取的是大小
            pixmap = QImage(rect.height(), rect.width(), QtGui.QImage.Format_ARGB32_Premultiplied)
            painter = QPainter(pixmap)
            # painter.begin(pixmap)
            rectf = QRectF(0, 0, pixmap.rect().height(), pixmap.rect().width())
            self.old_img.scene().render(painter, rectf, rect)
            painter.end()
            # 图像格式转换
            cv_img = self.QImage2CV(pixmap)  # numpy
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            net = load_net()
            cv_img = inference(net, gray_img)
            self.infer_image[self.currentImgIdx] = cv_img
            self.load_infer_Image()
        else:
            print('当前图像为空')
        # self.time.stop()
        # return

    def infer_all(self):  # 推理所有图像

        self.start_time = QDateTime.currentDateTime()
        self.time_q.timeout.connect(self.refresh)
        self.time_q.start()

        self.inferNet = loadNetThread()  # 创建一个线程
        self.inferNet.start()  #线程启动
        UIOperate.img_path_glob = self.image_paths
        self.inferNet.signal.connect(self.updateBar)

        #获取设置文本
        self.inferNet.organ.connect(self.setLable_16)
        self.inferNet.tumor.connect(self.setLabel_2)

        #接受图像
        self.inferNet.imgSignal.connect(self.showimg)

        # self.inferNet.stop()
        # net = load_net()
        # # # print(self.image_paths)  #'C:/Users/Enlong/OneDrive - 60knxp/学习资料/软件工程/版本更新/ui-project/image/39\\0.png'等
        # for i in range(len(self.image_paths)):
        #     path = self.image_paths[i]
        #     img = cv2.imread(path)  #cv读取需要英文路径
        #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     img = inference(net, gray_img)
        #     self.infer_image[i] = img
        #     self.progressBar_2.setValue(int(100 * i / len(self.image_paths)))
        # print("heool")
        # self.progressBar_2.setValue(100)
        '''注意退出不要直接退出，要使用step按钮结束推理后再退出，在推理中退出好像只是退出主窗口，推理还在运行'''


    def updateBar(self, i):
        # print(len(self.image_paths))   #90
        # print(i)  #1~90
        self.progressBar_2.setValue(int(100 * i / len(self.image_paths)))
        if i == len(self.image_paths):  #可以实现
            self.time_q.stop()

    def setLable_16(self, organ_size):
        self.label_16.setText(str(organ_size))

    def setLabel_2(self, tumor_size):
        self.label_2.setText(str(tumor_size))

    def showimg(self, img):
        self.img3D = img
        # print(type(img))  #numpy.ndarray
        print(img.shape)
        # c = img.copy()
        np.squeeze(img)
        np.resize(img, (16, 16))   #不会发生变化???
        # print(img.shape)
        # print(img)

    # def dimtothree(self, onedim, size, number):
    #     threedim = onedim.reshape()
    def concat_along_new_dim(self, array_list):
        temp = []
        for item in array_list:
            # print(type(item))
            # print(item)
            temp.append(item[np.newaxis, :])
        return np.concatenate(temp, axis=0)

    def view3D(self):  # 预测结果3D模型展示   d，h，w
        # self.getimg = loadNetThread()  #实例化
        # print("********************************")
        # print(self.getimg.img_glob)

        # matrix = np.random.randint(0, 10, size=(16, 16, 30))
        array_list = []
        #################只需将img3D改变形状即可,但是不知道为什么改变不了
        for i in range(0, 3):
            # array_list.append(np.random.randint(low=10, high=100, size=[1080, 1920]))
            array_list.append(self.img3D)
        matrix = self.concat_along_new_dim(array_list)
        # a = np.random.randint(0, 10, size=[3,4])
        # print(a)
        # print(type(a))
        # print(a.shape)
        # print(self.img3D)   ########要读取数据不是None,否则报错
        # print(self.img3D.shape)
        # print(type(self.img3D))
        # matrix = np.array([self.img3D, self.img3D,self.img3D])  #可以获取到
        # matrix = np.ones((512, 512, 90))
        self.viewer = Matrix3DViewer(matrix)

        self.viewer.show()
        self.viewer.plot_matrix()
        # xy = []
        # z = []
        # for i in range(len(self.image_paths)):
        #     z.append(i)
        #     xy.append(self.infer_image[i][:][:])
        # self.viewer.axes.scatter(xy[:, 0], xy[:, 1], z, c=self.viewer.matrix[xy[:, 0], xy[:, 1], z])
        # self.viewer.canvas.draw()
        # self.viewer.show()

    def refresh(self):
        self.lcdNumber.setDigitCount(12)
        self.lcdNumber.setMode(QLCDNumber.Dec)
        curtime = QDateTime.currentDateTime()
        interval = self.start_time.msecsTo(curtime)
        self.lcdNumber.display(interval / 1000)


    def exitNet(self):  # 不知道可不可以直接退出——这个直接退出整个系统
        # sys.exit()  # 需要暂停的是网络而不是整个系统
        self.inferNet.stop()
        self.time_q.stop()   #不能重新启动
        self.progressBar_2.reset()

class loadNetThread(QThread):
    signal = pyqtSignal(int)  #int
    organ = pyqtSignal(int)
    tumor = pyqtSignal(int)
    imgSignal = pyqtSignal(np.ndarray)   #也可以，使用object可以传输所有类型数据

    # img_glob = np.zeros((512, 512))  #直接固定了
    def __init__(self):
        super(loadNetThread, self).__init__()  #这个里面用加的吗，不一样
        self.infer = UIOperate()   #可以吗
        self.flag = 1
        self.organ_size_all = 0
        self.tumor_size_all = 0
        self.imgArr = None

    def run(self):  #重写QThread类方法
        # self.flag = 1
        if self.flag == 1:
            net = load_net()
            # print(self.infer.img_path_glob)  #获取不到，是空的
            for i in range(len(self.infer.img_path_glob)):
                path = self.infer.img_path_glob[i]
                img = cv2.imread(path)  # cv读取需要英文路径
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = inference(net, gray_img)
                self.infer.infer_image[i] = img
                # loadNetThread.img_glob = new_img
                # print(img.shape)  #512*512

                self.signal.emit(i+1)
                self.organ_size_all += np.sum(img == 1)
                self.tumor_size_all += np.sum(img == 2)
                # new_img = img.astype("uint8")
                # cv2.resize(new_img, (16, 16), cv2.INTER_AREA)  #???不能
                # print(new_img)
                self.imgArr = img
                # np.resize(self.imgArr, (16, 16))
                # print(self.imgArr)
                # print(type(self.imgArr))  #<class 'numpy.ndarray'>
                # print(np.shape(self.imgArr))
        # self.organ_size_all += 111
        # self.tumor_size_all += 222
        self.organ.emit(self.organ_size_all)
        self.tumor.emit(self.tumor_size_all)
        self.imgSignal.emit(self.imgArr)
        # self.label_21.setText(str(organ_size_all))
        # self.label_22.setText(str(tumor_size_all))

    def stop(self):     #重写stop方法——有问题，退出一次，其它的不会退出，还需要改一下
        self.flag = 0														# 2.
        print('线程退出')


# 弹出窗体类
# 第一个案例的貌似直接使用QMessage.about就可以——查看评论的时候显示的
class Widget_Test(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(Widget_Test, self).__init__()
        self.setupUi(self)

    def showwidget(self):
        self.show()


if __name__ == '__main__':
    # app = QApplication([])
    app = QApplication(sys.argv)
    img = UIOperate()
    img.show()
    app.exec_()
