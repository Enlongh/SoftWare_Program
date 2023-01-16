import os
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon,QPixmap
from PyQt5.QtWidgets import QListWidget,QListWidgetItem,QListView,QWidget,QApplication,QHBoxLayout,QLabel

class ImageListWidget(QListWidget):
    def __init__(self):
        super(ImageListWidget, self).__init__()
        self.setFlow(QListView.Flow(1))#0: left to right,1: top to bottom
        self.setIconSize(QSize(150,100))

    def add_image_items(self,image_paths=[]):
        for img_path in image_paths:
            if os.path.isfile(img_path):
                img_name = os.path.basename(img_path)
                item = QListWidgetItem(QIcon(img_path),img_name)
                # item.setText(img_name)
                # item.setIcon(QIcon(img_path))
                self.addItem(item)


class ImageViewerWidget(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        # 显示控件
        self.list_widget = ImageListWidget()
        self.list_widget.setMinimumWidth(200)
        self.show_label = QLabel(self)
        self.show_label.setFixedSize(600,400)
        self.image_paths = []
        self.currentImgIdx = 0
        self.currentImg = None

        # 水平布局
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.show_label)
        self.layout.addWidget(self.list_widget)

        # 信号与连接
        self.list_widget.itemSelectionChanged.connect(self.loadImage)

    def load_from_paths(self,img_paths=[]):
        self.image_paths = img_paths
        self.list_widget.add_image_items(img_paths)

    def loadImage(self):
        self.currentImgIdx = self.list_widget.currentIndex().row()
        if self.currentImgIdx in range(len(self.image_paths)):
            self.currentImg = QPixmap(self.image_paths[self.currentImgIdx]).scaledToHeight(400)
            self.show_label.setPixmap(self.currentImg)
