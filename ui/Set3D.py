import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# class Matrix3DViewer(QWidget):
class Matrix3DViewer(QMainWindow):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.setCentralWidget(self.canvas)
        # self.canvas.mpl_connect('scroll_event', self.on_mouse_wheel)  #添加鼠标滚动事件监听器
        # self.zoom_factor = 1.0  # 初始化放大倍数
        self.setWindowTitle("结果的3D模型")

    def plot_matrix(self):
        x, y, z = self.matrix.nonzero()
        self.axes.scatter(x, y, z, c=self.matrix[x, y, z])
        self.canvas.draw()

    # def on_mouse_wheel(self, event):
    #     # 计算新的放大倍数
    #     factor = 1.1 if event.button == 'up' else 0.9
    #     self.zoom_factor *= factor
    #
    #     # 调用 set_zoom 方法进行视角放大
    #     self.axes.set_zoom(self.zoom_factor)
    #
    #
    #     # 刷新图像
    #     self.draw()


'''
        # 添加鼠标滚动事件监听器
        self.mpl_connect('scroll_event', self.on_mouse_wheel)

        # 初始化放大倍数
        self.zoom_factor = 1.0

    def on_mouse_wheel(self, event):
        # 计算新的放大倍数
        factor = 1.1 if event.button == 'up' else 0.9
        self.zoom_factor *= factor

        # 调用 set_zoom 方法进行视角放大
        self.axes.set_zoom(self.zoom_factor)

        # 刷新图像
        self.draw()
'''

# def on_click(self, event):  #不知道干什么——复位？？
#     if event.button == 1:
#         self.axes.view_init(elev=10, azim=10)
#     elif event.button == 3:
#         self.axes.view_init(elev=0, azim=0)
#     self.canvas.draw()


# if __name__ == '__main__':
#     matrix = np.random.randint(0, 10, size=(10, 10, 10))
#     app = QApplication(sys.argv)
#     viewer = Matrix3DViewer(matrix)
#     viewer.show()
#     viewer.plot_matrix()
#     app.exec_()
#     sys.exit()
