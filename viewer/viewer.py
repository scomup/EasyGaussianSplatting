#!/usr/bin/env python3

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QWidget, QComboBox, QVBoxLayout, QSizePolicy,\
      QSpacerItem, QLabel, QLineEdit, QMainWindow, QApplication
from OpenGL.GL import *
from PyQt5.QtGui import QKeyEvent, QIntValidator, QDoubleValidator, QVector3D


class SettingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.combo_items = QComboBox()
        self.combo_items.currentIndexChanged.connect(self.on_combobox_selection)
        self.layout = QVBoxLayout()
        self.stretch = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(self.stretch)
        self.layout.addWidget(self.combo_items)
        self.setLayout(self.layout)
        self.setWindowTitle("Setting Window")
        self.setGeometry(200, 200, 300, 200)
        self.items = None
        self.tmpWidgets = []

    def addComboItems(self, items):
        self.items = items
        for name, item in items.items():
            self.combo_items.addItem("%s(%s)" % (name, item.__class__.__name__))

    def on_combobox_selection(self, index):
        self.layout.removeItem(self.stretch)
        for w in self.tmpWidgets:
            self.layout.removeWidget(w)
        self.tmpWidgets = []

        key = list(self.items.keys())
        try:
            settings = self.items[key[index]].settings
            for setting in settings:
                label = QLabel(setting["name"])
                self.layout.addWidget(label)
                line_edit = QLineEdit()
                self.layout.addWidget(label)
                self.layout.addWidget(line_edit)
                self.tmpWidgets.append(label)
                self.tmpWidgets.append(line_edit)
                line_edit.textChanged.connect(lambda txt, setting=setting: self.on_change_setting(txt, setting))
                line_edit.setText(str(setting["get"]()))
                if(setting["type"] == np.int32 or setting["type"] == np.int64 or setting["type"] == int):
                    line_edit.setValidator(QIntValidator())
                elif(setting["type"] == np.float32 or setting["type"] == np.float64 or setting["type"] == float):
                    line_edit.setValidator(QDoubleValidator())
        except:
            print("No settings")
        self.layout.addItem(self.stretch)

    def on_change_setting(self, text, setting):
        try:
            input = setting["type"](text)
            setting["set"](input)
        except:
            pass


class MyViewWidget(gl.GLViewWidget):
    def __init__(self):
        super(MyViewWidget, self).__init__()
        self.setting_window = SettingWindow()
        self.plugin_items = None

    def paintGL(self, region=None, viewport=None, useItemNames=False):
        """
        viewport specifies the arguments to glViewport. If None, then we use self.opts['viewport']
        region specifies the sub-region of self.opts['viewport'] that should be rendered.
        Note that we may use viewport != self.opts['viewport'] when exporting.
        """
        # print("in")
        if viewport is None:
            glViewport(*self.getViewport())
        else:
            glViewport(*viewport)
        self.setProjection(region=region)

        camera_postion = np.array(self.viewMatrix().copyDataTo()).reshape(4, 4)
        self.setModelview()
        bgcolor = self.opts['bgcolor']
        glClearColor(*bgcolor)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
        self.drawItemTree(useItemNames=useItemNames)

    def addPluginItems(self, items):
        self.plugin_items = items
        for item in items.values():
            self.addItem(item)
        self.setting_window.addComboItems(self.plugin_items)

    def mouseReleaseEvent(self, ev):
        if hasattr(self, 'mousePos'):
            delattr(self, 'mousePos')

    def mouseMoveEvent(self, ev):
        lpos = ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos
        if ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            pitch_abs = np.abs(self.opts['elevation'])
            camera_mode = 'view-upright'
            if(pitch_abs <= 45.0 or pitch_abs == 90):
                camera_mode = 'view'
            self.pan(diff.x(), diff.y(), 0, relative=camera_mode)

    def follow(self, p):
        self.setCameraPosition(pos=QVector3D(p[0], p[1], p[2]))

    def keyPressEvent(self, event: QKeyEvent):

        if event.key() == QtCore.Qt.Key_M:  # setting meun
            print("Open setting windows")
            self.open_setting_window()

        else:
            super().keyPressEvent(event)

    def open_setting_window(self):
        self.setting_window.show()


class Viewer(QMainWindow):
    def __init__(self, items):
        self.items = items
        super(Viewer, self).__init__()
        self.setGeometry(0, 0, 1080, 720)
        self.initUI()
        self.setWindowTitle("Fast Viewer")

    def open_setting_window(self):
        self.setting_window = SettingWindow()
        self.setting_window.show()

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)
        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = MyViewWidget()
        layout.addWidget(self.viewer, 1)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.update)

        self.viewer.setCameraPosition(distance=40)

        g = gl.GLGridItem()
        g.setSize(50, 50)
        g.setSpacing(1, 1)
        self.viewer.addItem(g)

        self.viewer.addPluginItems(self.items)

        timer.start()

    def update(self):
        self.viewer.update()


if __name__ == '__main__':
    app = QApplication([])
    viewer = Viewer({})
    viewer.show()
    app.exec_()
