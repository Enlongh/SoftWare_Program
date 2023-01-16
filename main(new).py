import numpy as np
from PyQt5.QtWidgets import (QApplication, QListWidgetItem, QMainWindow, QFileDialog, QGraphicsItem,
                             QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsView, QInputDialog, QLCDNumber)
from PyQt5.QtGui import QPixmap, QImage, QIcon, qRed, qBlue, qGreen, qRgb
from PyQt5.QtCore import QTimer, QDateTime, QSize
from ui.main_win import *
from ImageLoad.imageview import *
import os
import cv2
from ui.sub_window import *


# from data_utils.inference import *
# from data_utils.visualize import *


class UIOperate(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(UIOperate, self).__init__(parent)
        self.setupUi(self)

        self.read_file.triggered.connect(self.load_from_paths)  # openimage
        self.save_file.triggered.connect(self.saveimage)

        self.currentImgIdx = 0
        self.image_paths = []
        self.infer_image = [0] * 100

        # 所以这个展示图像的功能是干什么用的？？
        self.Button_display_oldimg.clicked.connect(self.load_infer_Image)
        self.imgList.itemSelectionChanged.connect(self.loadImage)

        # self.view = QGraphicsView(self)
        # self.view.setGeometry(0, 0, 710, 650)
        self.graphicsScene = QGraphicsScene()  # 创建场景
        self.result_graphicsScene = QGraphicsScene()  # 创建场景
        self.imgFile = None

        # 图像处理
        self.actionGaussian.triggered.connect(self.Gaussianblur)
        self.actionBrightnessContrast.triggered.connect(self.Hist)
        self.actionGamma.triggered.connect(self.Gammatransform)
        self.actionNoneTranspose.triggered.connect(self.Recover)

        self.action_edit_comments.triggered.connect(self.editcomments)
        self.commentpath = r'C:\Users\Enlong\Desktop\commnet.txt'  # 考虑可以选择保存路径

        self.action_view_Comments.triggered.connect(self.viewcomments)
        self.sub_window = Widget_Test()

        '''推理操作'''
        self.infer_2.clicked.connect(self.infer_current)  # 提取当前帧
        self.pushButton_3.clicked.connect(self.infer_all)  # 提取所有帧
        # 两个操作是不是有冲突
        self.pushButton_10.clicked.connect(self.exitNet)
        self.lineEdit.setText('-150')
        self.lineEdit_2.setText('350')
        self.lineEdit.returnPressed.connect(self.cut_window)
        self.lineEdit_2.returnPressed.connect(self.cut_window)
        self.time = QTimer(self)
        self.time.setInterval(100)
        self.start_time = 0
        # self.time.timeout.connect(self.refresh)

    def cut_window(self):  # 更新窗位大小
        window_min = self.lineEdit.text()
        window_max = self.lineEdit_2.text()
        window = "[" + window_min + " " + window_max + "]"
        self.label_16.setText(window)

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

                pix = qRgb(r, g, b)
                qimg.setPixel(col, row, pix)
        return qimg  # 转换完成，返回

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
            self.textBrowser_2.setPlainText(str(self.image_paths[self.currentImgIdx]))
            pixmapItem = QGraphicsPixmapItem(self.currentImg)  # 创建像素图元
            self.graphicsScene.addItem(pixmapItem)
            self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
            self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像
            self.old_img.setScene(self.graphicsScene)
            self.old_img.show()
            self.label_10.setText(str(self.currentImgIdx + 1))
            self.label_15.setText(str(self.currentImg.height()))

    def load_infer_Image(self):  # 加载推理图像
        self.currentImgIdx = self.imgList.currentIndex().row()
        if self.currentImgIdx in range(len(self.image_paths)):
            img = self.infer_image[self.currentImgIdx]
            img = img * 127.5
            img = img.astype("uint8")
            print(img)
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

    def Gammatransform(self):
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

    def Recover(self):
        img_path = self.textBrowser_2.toPlainText()
        # print(img_path)
        pix = QPixmap(img_path)
        pixmapItem = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.graphicsScene.addItem(pixmapItem)
        self.old_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭滚动条
        self.old_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不关闭滚动条会挡住图像

        self.old_img.setScene(self.graphicsScene)
        self.old_img.show()

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
        rect = self.old_img.scene().sceneRect()  # 这个获取的是大小
        pixmap = QImage(rect.height(), rect.width(), QtGui.QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(pixmap)
        painter.begin(pixmap)
        rectf = QRectF(0, 0, pixmap.rect().height(), pixmap.rect().width())
        self.old_img.scene().render(painter, rectf, rect)
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
        self.sub_window.text_browser.setText(text)
        self.sub_window.showwidget()

    '''推理'''

    def infer_current(self):
        self.start_time = QDateTime.currentDateTime()
        self.time.timeout.connect(self.refresh)
        self.time.start()
        # path = self.image_paths[self.currentImgIdx]
        # img = cv2.imread(path)
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # net = load_net()
        # img = inference(net, gray_img)
        # self.infer_image[self.currentImgIdx] = img
        # self.time.stop()

        '''推理进度怎么算？'''
        for i in range(101):
            self.progressBar_2.setValue(i)  # 进度条当前值
        return

    def infer_all(self):  # 推理所有图像
        # start_time = time.time()
        self.start_time = QDateTime.currentDateTime()
        # print("hello",self.start_time)
        self.time.timeout.connect(self.refresh)
        self.time.start()
        net = load_net()
        for i in range(len(self.image_paths)):
            path = self.image_paths[i]
            img = cv2.imread(path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = inference(net, gray_img)
            self.infer_image[i] = img
            print(i)
        self.time.stop()

    def refresh(self):
        self.lcdNumber.setDigitCount(12)
        self.lcdNumber.setMode(QLCDNumber.Dec)

        curtime = QDateTime.currentDateTime()
        interval = self.start_time.msecsTo(curtime)
        self.lcdNumber.display(interval / 1000)

    def exitNet(self):  # 不知道可不可以直接退出——这个直接退出整个系统
        sys.exit()  # 需要暂停的是网络而不是整个系统


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
