from PyQt5.QtWidgets import QTabWidget,QAction, QFileDialog
from PyQt5.QtGui import  QColor


import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QLineEdit, QMessageBox
from math import cos, sin, radians

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import numpy as np
from PIL import Image

from GUI.Odev1Page import Odev1Page
from GUI.Odev2Page import Odev2Page
from GUI.VizePage import VizePage



class ImageProcessingWindow(QWidget):
    def __init__(self, image, parent=None):
        super().__init__(parent)

        # Görüntüyü göstermek için bir QLabel oluştur
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap.fromImage(image))

        # Histogram ve kanallar için etiketler oluştur
        self.histogram_label = QLabel("Histogram")
        self.channels_label = QLabel("Kanallar")

        # Layout oluştur ve widget'ları layouta ekle
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.histogram_label)
        layout.addWidget(self.channels_label)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.histogram_tab = None  # histogram_tab değişkenini tanımlıyoruz

        # Ana pencere özelliklerini ayarla
        self.setWindowTitle("Dijital Görüntü İşleme")
        self.setGeometry(200, 200, 1000, 1000)

        # Ana widget oluştur
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Ana layout oluştur
        layout = QVBoxLayout(main_widget)

        # Başlık etiketi oluştur ve ana layouta ekle
        title_label = QLabel("Dijital Görüntü İşleme Uygulaması", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #344955; margin: 30px;")
        layout.addWidget(title_label)

        # Öğrenci bilgi etiketi oluştur ve ana layouta ekle
        student_info_label = QLabel("Numara: 2112290034\nAd Soyad: Esra Tosun", self)
        student_info_label.setAlignment(Qt.AlignCenter)
        student_info_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #50727B; margin: 15px;")
        layout.addWidget(student_info_label)

        # Arka plan rengini ayarla
        background_color = QColor("#78A083")
        main_widget.setStyleSheet(f"background-color: {background_color.name()};")

        # Ana pencere için sekme widget'ı oluştur ve ana layouta ekle
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Sekmeler oluştur
        self.create_odev1_page()
        self.create_odev2_page()
        self.create_vize_page()
        self.create_histogram_page()

        # Menüleri oluştur
        self.create_actions()
        self.create_menu_navigation()

    def create_histogram_page(self):
        # Histogram sekmesi için bir QTabWidget oluştur
        self.histogram_tab = QTabWidget()

        # Ana tablo widget'ına histogram sekmesini ekleyin
        self.tab_widget.addTab(self.histogram_tab, "Histogram")

    def create_odev1_page(self):
        # Ana sekme oluştur ve sekme widget'ına ekle
        odev1_tab = Odev1Page()
        self.tab_widget.addTab(odev1_tab, "Ödev 1: Temel İşlevselliği Oluştur")

    def create_odev2_page(self):
        # Ödev2 sekmesini oluştur ve sekme widget'ına ekle
        odev2_tab = Odev2Page()
        self.tab_widget.addTab(odev2_tab, "Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon")

    def create_vize_page(self):
        # Ödev2 sekmesini oluştur ve sekme widget'ına ekle
        vize = VizePage()
        self.tab_widget.addTab(vize, "Vize")

    def create_actions(self):
        # Yeni eylem oluştur ve tetikleyici ataması yap
        self.new_action = QAction("Yeni", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.triggered.connect(self.new_image_processing_window)

    def create_menu_navigation(self):
        # Ana menü çubuğunu oluştur
        menubar = self.menuBar()
        menubar.setStyleSheet("background-color: #f0f0f0;")

        # Dosya menüsünü oluştur ve eylemi ekle
        file_menu = menubar.addMenu("Dosya")
        file_menu.addAction(self.new_action)

    def show_home_page(self):
        # Ana sayfaya git
        self.tab_widget.setCurrentIndex(0)

    def new_image_processing_window(self):
        # Yeni bir görüntü işleme penceresi oluştur
        image = QImage(640, 480, QImage.Format_RGB32)
        image.fill(Qt.white)  # Beyaz bir arka plan ekleyin
        image_processing_window = ImageProcessingWindow(image)

        # Histogram sekmesine yeni bir sayfa ekle
        self.histogram_tab.addTab(image_processing_window, "Yeni İşlem")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
