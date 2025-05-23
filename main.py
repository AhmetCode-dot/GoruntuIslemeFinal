import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMenuBar, QAction, QMenu, QMessageBox, QSizePolicy,
                            QSpinBox, QComboBox, QDoubleSpinBox, QProgressBar,
                            QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
import math
import os
import pandas as pd

# Global stil tanımlamaları
STYLE_SHEET = """
QMainWindow {
    background-color: #f0f0f0;
}

QMenuBar {
    background-color: #2c3e50;
    min-height: 60px;
    font-size: 22px;
    color: white;
}

QMenuBar::item {
    padding: 15px 30px;
    margin: 5px;
    border-radius: 8px;
    font-weight: bold;
}

QMenuBar::item:selected {
    background-color: #3498db;
}

QMenu {
    background-color: #2c3e50;
    color: white;
    border: none;
    padding: 8px;
    border-radius: 8px;
}

QMenu::item {
    padding: 12px 25px;
    border-radius: 5px;
    font-size: 16px;
}

QMenu::item:selected {
    background-color: #3498db;
}

QLabel {
    color: #2c3e50;
    font-size: 14px;
}

QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 14px;
    min-width: 120px;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:disabled {
    background-color: #bdc3c7;
}

QComboBox {
    background-color: white;
    border: 2px solid #bdc3c7;
    border-radius: 5px;
    padding: 5px;
    min-width: 120px;
    font-size: 14px;
}

QComboBox:hover {
    border-color: #3498db;
}

QSpinBox, QDoubleSpinBox {
    background-color: white;
    border: 2px solid #bdc3c7;
    border-radius: 5px;
    padding: 5px;
    font-size: 14px;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #3498db;
}

QProgressBar {
    border: 2px solid #bdc3c7;
    border-radius: 5px;
    text-align: center;
    background-color: white;
}

QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 3px;
}

ImageFrame {
    background-color: white;
    border: 2px solid #bdc3c7;
    border-radius: 10px;
    padding: 10px;
}
"""

class ImageFrame(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("ImageFrame")
        layout = QVBoxLayout(self)
        
        # Başlık
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Görüntü alanı
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid #bdc3c7; border-radius: 5px;")
        layout.addWidget(self.image_label)

class ImageProcessingThread(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, image, operation, factor, angle, interpolation):
        super().__init__()
        self.image = image
        self.operation = operation
        self.factor = factor
        self.angle = angle
        self.interpolation = interpolation
        
    def run(self):
        try:
            if self.operation == "Büyütme":
                result = self.resize_image(self.image, self.factor, self.interpolation)
            elif self.operation == "Küçültme":
                result = self.resize_image(self.image, 1/self.factor, self.interpolation)
            elif self.operation == "Zoom In":
                result = self.zoom_image(self.image, self.factor, self.interpolation, True)
            elif self.operation == "Zoom Out":
                result = self.zoom_image(self.image, self.factor, self.interpolation, False)
            elif self.operation == "Döndürme":
                result = self.rotate_image(self.image, self.angle, self.interpolation)
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def resize_image(self, image, factor, interpolation):
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        return self.apply_interpolation(image, new_h, new_w, interpolation)

    def zoom_image(self, image, factor, interpolation, is_zoom_in):
        try:
            h, w = image.shape[:2]
            center_h, center_w = h // 2, w // 2
            
            if is_zoom_in:
                # Zoom In: Merkezden küçük bir alan seçip büyüt
                new_h, new_w = int(h / factor), int(w / factor)
                start_h = max(0, center_h - new_h // 2)
                start_w = max(0, center_w - new_w // 2)
                end_h = min(h, start_h + new_h)
                end_w = min(w, start_w + new_w)
                
                cropped = image[start_h:end_h, start_w:end_w]
                return self.apply_interpolation(cropped, h, w, interpolation)
            else:
                # Zoom Out: Görüntüyü küçült ve siyah kenarlık ekle
                # Faktörü 1'den küçük yaparak küçültme işlemi
                zoom_factor = min(1.0, 1.0 / factor)  # Faktörü 1'den küçük yap
                new_h = max(1, int(h * zoom_factor))
                new_w = max(1, int(w * zoom_factor))
                
                # Görüntüyü küçült
                resized = self.apply_interpolation(image, new_h, new_w, interpolation)
                
                # Siyah arka plan oluştur
                result = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Küçültülmüş görüntüyü merkeze yerleştir
                start_h = (h - new_h) // 2
                start_w = (w - new_w) // 2
                
                # Sınırları kontrol et
                if start_h >= 0 and start_w >= 0 and start_h + new_h <= h and start_w + new_w <= w:
                    result[start_h:start_h + new_h, start_w:start_w + new_w] = resized
                else:
                    # Eğer sınırlar aşılıyorsa, görüntüyü kırp
                    valid_h = min(new_h, h - start_h)
                    valid_w = min(new_w, w - start_w)
                    if valid_h > 0 and valid_w > 0:
                        result[start_h:start_h + valid_h, start_w:start_w + valid_w] = resized[:valid_h, :valid_w]
                
                return result
        except Exception as e:
            print(f"Zoom hatası: {str(e)}")
            raise Exception(f"Zoom işlemi sırasında hata oluştu: {str(e)}")

    def rotate_image(self, image, angle, interpolation):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
        return self.apply_interpolation(rotated, h, w, interpolation)

    def apply_interpolation(self, image, new_h, new_w, interpolation):
        if interpolation == "Bilinear":
            return self.bilinear_interpolation(image, new_h, new_w)
        elif interpolation == "Bicubic":
            return self.bicubic_interpolation(image, new_h, new_w)
        else:  # Average
            return self.average_interpolation(image, new_h, new_w)

    def bilinear_interpolation(self, image, new_h, new_w):
        h, w = image.shape[:2]
        result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        for i in range(new_h):
            for j in range(new_w):
                x = (j + 0.5) * w / new_w - 0.5
                y = (i + 0.5) * h / new_h - 0.5
                
                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                
                wx = x - x1
                wy = y - y1
                
                result[i, j] = (1 - wx) * (1 - wy) * image[y1, x1] + \
                              wx * (1 - wy) * image[y1, x2] + \
                              (1 - wx) * wy * image[y2, x1] + \
                              wx * wy * image[y2, x2]
        
        return result

    def bicubic_interpolation(self, image, new_h, new_w):
        h, w = image.shape[:2]
        result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        def cubic_weight(x):
            x = abs(x)
            if x < 1:
                return 1.5 * x**3 - 2.5 * x**2 + 1
            elif x < 2:
                return -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
            return 0
        
        for i in range(new_h):
            for j in range(new_w):
                x = (j + 0.5) * w / new_w - 0.5
                y = (i + 0.5) * h / new_h - 0.5
                
                x1, y1 = int(x), int(y)
                pixels = []
                weights = []
                
                for dy in range(-1, 3):
                    for dx in range(-1, 3):
                        nx, ny = x1 + dx, y1 + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            pixels.append(image[ny, nx])
                            weights.append(cubic_weight(x - nx) * cubic_weight(y - ny))
                
                if pixels:
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)
                    result[i, j] = np.sum(pixels * weights[:, np.newaxis], axis=0)
        
        return result

    def average_interpolation(self, image, new_h, new_w):
        h, w = image.shape[:2]
        result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        for i in range(new_h):
            for j in range(new_w):
                x = (j + 0.5) * w / new_w - 0.5
                y = (i + 0.5) * h / new_h - 0.5
                
                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                
                result[i, j] = np.mean([
                    image[y1, x1],
                    image[y1, x2],
                    image[y2, x1],
                    image[y2, x2]
                ], axis=0)
        
        return result

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dijital Görüntü İşleme")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet(STYLE_SHEET)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        self.header = QLabel("Dijital Görüntü İşleme\nÖğrenci No: 231229068\nAd Soyad: AHMET BAĞRIYANIK")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont('Arial', 24, QFont.Bold))
        self.header.setStyleSheet("""
            color: #2c3e50;
            margin: 20px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            border: 2px solid #bdc3c7;
        """)
        self.layout.addWidget(self.header)
        
        # Ödevler butonu için container
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setAlignment(Qt.AlignCenter)
        
        # Ödevler butonu
        self.homework_button = QPushButton("Ödevler")
        self.homework_button.setFont(QFont('Arial', 18, QFont.Bold))
        self.homework_button.setMinimumSize(200, 60)
        self.homework_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        # Ödevler menüsü
        self.homework_menu = QMenu(self)
        self.homework_menu.setStyleSheet("""
            QMenu {
                background-color: #2c3e50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 8px;
            }
            QMenu::item {
                padding: 12px 25px;
                border-radius: 5px;
                font-size: 16px;
            }
            QMenu::item:selected {
                background-color: #3498db;
            }
        """)
        
        # Ödev 1
        hw1_action = QAction('Ödev 1: Temel İşlevselliği Oluştur', self)
        hw1_action.triggered.connect(self.show_homework1)
        self.homework_menu.addAction(hw1_action)
        
        # Ödev 2
        hw2_action = QAction('Ödev 2: Fotoğrafta Görüntü Operasyonları', self)
        hw2_action.triggered.connect(self.show_homework2)
        self.homework_menu.addAction(hw2_action)
        
        # Final Ödevi
        final_action = QAction('Final Ödevi: Görüntü İşleme Operasyonları', self)
        final_action.triggered.connect(self.show_final_project)
        self.homework_menu.addAction(final_action)
        
        # Butona menüyü bağla
        self.homework_button.clicked.connect(self.show_homework_menu)
        
        button_layout.addWidget(self.homework_button)
        self.layout.addWidget(button_container)
        
        # Boş alan ekleyerek butonu yukarıda tut
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(spacer)

    def show_homework_menu(self):
        # Butonun altında menüyü göster
        self.homework_menu.exec_(self.homework_button.mapToGlobal(
            self.homework_button.rect().bottomLeft()))

    def show_homework1(self):
        self.homework1_window = Homework1Window()
        self.homework1_window.show()

    def show_homework2(self):
        self.homework2_window = Homework2Window()
        self.homework2_window.show()

    def show_final_project(self):
        self.final_project_window = FinalProjectWindow()
        self.final_project_window.show()

class Homework1Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ödev 1: Temel İşlevsellik")
        self.setGeometry(150, 150, 1000, 700)
        self.setStyleSheet(STYLE_SHEET)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        self.header = QLabel("Dijital Görüntü İşleme - Ödev 1\nÖğrenci No: 231229068\nAd Soyad: AHMET BAĞRIYANIK")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont('Arial', 16, QFont.Bold))
        self.header.setStyleSheet("""
            color: #2c3e50;
            margin: 10px;
            padding: 15px;
            background-color: white;
            border-radius: 10px;
            border: 2px solid #bdc3c7;
        """)
        self.layout.addWidget(self.header)
        
        # Görüntü gösterme alanı
        image_frame = QFrame()
        image_frame.setObjectName("ImageFrame")
        image_layout = QVBoxLayout(image_frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 400)
        self.image_label.setStyleSheet("""
            background-color: white;
            border: 2px solid #bdc3c7;
            border-radius: 10px;
            padding: 10px;
        """)
        image_layout.addWidget(self.image_label)
        self.layout.addWidget(image_frame)
        
        # Butonlar için frame
        button_frame = QFrame()
        button_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        button_layout = QHBoxLayout(button_frame)
        button_layout.setSpacing(15)
        
        # Görüntü yükleme butonu
        self.load_button = QPushButton("Görüntü Yükle")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)
        
        # Gri tonlama butonu
        self.grayscale_button = QPushButton("Gri Tonlamaya Çevir")
        self.grayscale_button.clicked.connect(self.convert_to_grayscale)
        button_layout.addWidget(self.grayscale_button)
        
        # Parlaklık artırma butonu
        self.brightness_button = QPushButton("Parlaklığı Artır")
        self.brightness_button.clicked.connect(self.increase_brightness)
        button_layout.addWidget(self.brightness_button)
        
        # Kaydetme butonu
        self.save_button = QPushButton("Görüntüyü Kaydet")
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)
        
        self.layout.addWidget(button_frame)
        
        # Görüntü verisi
        self.current_image = None
        self.processed_image = None

    def load_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç",
                                                     "", "Image Files (*.png *.jpg *.jpeg)")
            if file_name:
                file_name = str(file_name)
                self.current_image = cv2.imdecode(
                    np.fromfile(file_name, dtype=np.uint8), 
                    cv2.IMREAD_UNCHANGED
                )
                
                if self.current_image is None:
                    raise Exception("Görüntü yüklenemedi")
                    
                self.display_image(self.current_image)
                
        except Exception as e:
            print(f"Hata: {str(e)}")
            QMessageBox.critical(self, "Hata", 
                               "Görüntü yüklenirken bir hata oluştu!\nLütfen dosya adında Türkçe karakter olmadığından emin olun.")

    def display_image(self, image):
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

    def convert_to_grayscale(self):
        if self.current_image is not None:
            self.processed_image = cv2.cvtColor(
                self.current_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.cvtColor(
                self.processed_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image)

    def increase_brightness(self):
        if self.current_image is not None:
            self.processed_image = cv2.convertScaleAbs(
                self.current_image, alpha=1.2, beta=30)
            self.display_image(self.processed_image)

    def save_image(self):
        if self.processed_image is not None:
            try:
                file_name, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet",
                                                         "", "PNG Files (*.png);;JPEG Files (*.jpg)")
                if file_name:
                    is_success, im_buf_arr = cv2.imencode(".png", self.processed_image)
                    if is_success:
                        im_buf_arr.tofile(file_name)
                    else:
                        raise Exception("Görüntü kaydedilemedi")
            except Exception as e:
                print(f"Hata: {str(e)}")
                QMessageBox.critical(self, "Hata", 
                                   "Görüntü kaydedilirken bir hata oluştu!")

class Homework2Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ödev 2: Fotoğrafta Görüntü Operasyonları")
        self.setGeometry(150, 150, 1200, 800)
        self.setStyleSheet(STYLE_SHEET)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        self.header = QLabel("Dijital Görüntü İşleme - Ödev 2\nÖğrenci No: 231229068\nAd Soyad: AHMET BAĞRIYANIK")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont('Arial', 16, QFont.Bold))
        self.header.setStyleSheet("""
            color: #2c3e50;
            margin: 10px;
            padding: 15px;
            background-color: white;
            border-radius: 10px;
            border: 2px solid #bdc3c7;
        """)
        self.layout.addWidget(self.header)
        
        # Görüntü gösterme alanları için yatay layout
        image_layout = QHBoxLayout()
        image_layout.setSpacing(20)
        
        # Orijinal görüntü için alan
        self.original_frame = ImageFrame("Orijinal Görüntü")
        image_layout.addWidget(self.original_frame)
        
        # İşlenmiş görüntü için alan
        self.processed_frame = ImageFrame("İşlenmiş Görüntü")
        image_layout.addWidget(self.processed_frame)
        
        self.layout.addLayout(image_layout)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                height: 20px;
                text-align: center;
            }
        """)
        self.layout.addWidget(self.progress_bar)
        
        # Kontrol paneli
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        control_layout = QHBoxLayout(control_frame)
        control_layout.setSpacing(15)
        
        # Görüntü yükleme butonu
        self.load_button = QPushButton("Görüntü Yükle")
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)
        
        # İşlem seçimi
        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "Büyütme", "Küçültme", "Zoom In", "Zoom Out", "Döndürme"
        ])
        control_layout.addWidget(self.operation_combo)
        
        # Faktör girişi
        factor_layout = QHBoxLayout()
        factor_label = QLabel("Faktör:")
        self.factor_spin = QDoubleSpinBox()
        self.factor_spin.setRange(0.1, 10.0)
        self.factor_spin.setValue(1.0)
        self.factor_spin.setSingleStep(0.1)
        factor_layout.addWidget(factor_label)
        factor_layout.addWidget(self.factor_spin)
        control_layout.addLayout(factor_layout)
        
        # Açı girişi (döndürme için)
        angle_layout = QHBoxLayout()
        angle_label = QLabel("Açı:")
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(0, 360)
        self.angle_spin.setValue(0)
        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_spin)
        control_layout.addLayout(angle_layout)
        
        # İnterpolasyon yöntemi seçimi
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems([
            "Bilinear", "Bicubic", "Average"
        ])
        control_layout.addWidget(self.interpolation_combo)
        
        # Uygula butonu
        self.apply_button = QPushButton("Uygula")
        self.apply_button.clicked.connect(self.apply_operation)
        control_layout.addWidget(self.apply_button)
        
        # Kaydet butonu
        self.save_button = QPushButton("Kaydet")
        self.save_button.clicked.connect(self.save_image)
        control_layout.addWidget(self.save_button)
        
        self.layout.addWidget(control_frame)
        
        # Görüntü verisi
        self.current_image = None
        self.processed_image = None
        
        # İşlem seçimine göre kontrolleri güncelle
        self.operation_combo.currentIndexChanged.connect(self.update_controls)
        self.update_controls()

    def update_controls(self):
        operation = self.operation_combo.currentText()
        self.factor_spin.setVisible(operation in ["Büyütme", "Küçültme", "Zoom In", "Zoom Out"])
        self.angle_spin.setVisible(operation == "Döndürme")

    def load_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç",
                                                     "", "Image Files (*.png *.jpg *.jpeg)")
            if file_name:
                file_name = str(file_name)
                self.current_image = cv2.imdecode(
                    np.fromfile(file_name, dtype=np.uint8), 
                    cv2.IMREAD_UNCHANGED
                )
                
                if self.current_image is None:
                    raise Exception("Görüntü yüklenemedi")
                    
                self.display_image(self.current_image, self.original_frame.image_label)
                self.processed_image = None
                self.processed_frame.image_label.clear()
                
        except Exception as e:
            print(f"Hata: {str(e)}")
            QMessageBox.critical(self, "Hata", 
                               "Görüntü yüklenirken bir hata oluştu!")

    def display_image(self, image, label):
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label.width(), label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)

    def apply_operation(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyin!")
            return

        # Butonları devre dışı bırak
        self.apply_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # İlerleme çubuğunu göster
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Belirsiz ilerleme
        
        # İşlem parametrelerini al
        operation = self.operation_combo.currentText()
        factor = self.factor_spin.value()
        angle = self.angle_spin.value()
        interpolation = self.interpolation_combo.currentText()
        
        # İşlem thread'ini başlat
        self.thread = ImageProcessingThread(
            self.current_image.copy(),
            operation,
            factor,
            angle,
            interpolation
        )
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()

    def on_processing_finished(self, result):
        self.processed_image = result
        self.display_image(self.processed_image, self.processed_frame.image_label)
        
        # Butonları tekrar etkinleştir
        self.apply_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # İlerleme çubuğunu gizle
        self.progress_bar.setVisible(False)

    def on_processing_error(self, error_msg):
        QMessageBox.critical(self, "Hata", f"İşlem sırasında bir hata oluştu: {error_msg}")
        
        # Butonları tekrar etkinleştir
        self.apply_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # İlerleme çubuğunu gizle
        self.progress_bar.setVisible(False)

    def save_image(self):
        if self.processed_image is not None:
            try:
                file_name, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet",
                                                         "", "PNG Files (*.png);;JPEG Files (*.jpg)")
                if file_name:
                    is_success, im_buf_arr = cv2.imencode(".png", self.processed_image)
                    if is_success:
                        im_buf_arr.tofile(file_name)
                    else:
                        raise Exception("Görüntü kaydedilemedi")
            except Exception as e:
                print(f"Hata: {str(e)}")
                QMessageBox.critical(self, "Hata", 
                                   "Görüntü kaydedilirken bir hata oluştu!")

class FinalProjectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Final Ödevi: Görüntü İşleme Operasyonları")
        self.setGeometry(150, 150, 1200, 800)
        self.setStyleSheet(STYLE_SHEET)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        self.header = QLabel("Dijital Görüntü İşleme - Final Ödevi\nÖğrenci No: 231229068\nAd Soyad: AHMET BAĞRIYANIK")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont('Arial', 16, QFont.Bold))
        self.header.setStyleSheet("""
            color: #2c3e50;
            margin: 10px;
            padding: 15px;
            background-color: white;
            border-radius: 10px;
            border: 2px solid #bdc3c7;
        """)
        self.layout.addWidget(self.header)
        
        # Görüntü gösterme alanları için yatay layout
        image_layout = QHBoxLayout()
        image_layout.setSpacing(20)
        
        # Orijinal görüntü için alan
        self.original_frame = ImageFrame("Orijinal Görüntü")
        image_layout.addWidget(self.original_frame)
        
        # İşlenmiş görüntü için alan
        self.processed_frame = ImageFrame("İşlenmiş Görüntü")
        image_layout.addWidget(self.processed_frame)
        
        self.layout.addLayout(image_layout)
        
        # Kontrol paneli
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        control_layout = QHBoxLayout(control_frame)
        control_layout.setSpacing(15)
        
        # Görüntü yükleme butonu
        self.load_button = QPushButton("Görüntü Yükle")
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)
        
        # İşlem seçimi
        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "Standart Sigmoid",
            "Yatay Kaydırılmış Sigmoid",
            "Eğimli Sigmoid",
            "Özel Fonksiyon",
            "Yol Çizgisi Tespiti",
            "Göz Tespiti",
            "Koyu Yeşil Bölge Analizi (Excel Çıktısı)",
            "Deblur"
        ])
        control_layout.addWidget(self.operation_combo)
        
        # Sigmoid parametreleri
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 10.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setSingleStep(0.1)
        alpha_label = QLabel("Alpha:")
        control_layout.addWidget(alpha_label)
        control_layout.addWidget(self.alpha_spin)
        
        # Beta parametresi (yatay kaydırma için)
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(-128, 128)
        self.beta_spin.setValue(0)
        self.beta_spin.setSingleStep(1)
        beta_label = QLabel("Beta:")
        control_layout.addWidget(beta_label)
        control_layout.addWidget(self.beta_spin)
        
        # Kernel size ve angle kontrolleri (sadece deblurring için görünür olacak)
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(3, 51)
        self.kernel_size_spin.setValue(15)
        self.kernel_size_spin.setSingleStep(2)
        kernel_size_label = QLabel("Kernel Boyutu:")
        control_layout.addWidget(kernel_size_label)
        control_layout.addWidget(self.kernel_size_spin)
        
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(0, 180)
        self.angle_spin.setValue(0)
        angle_label = QLabel("Açı:")
        control_layout.addWidget(angle_label)
        control_layout.addWidget(self.angle_spin)
        
        # Uygula butonu
        self.apply_button = QPushButton("Uygula")
        self.apply_button.clicked.connect(self.apply_operation)
        control_layout.addWidget(self.apply_button)
        
        # Kaydet butonu
        self.save_button = QPushButton("Kaydet")
        self.save_button.clicked.connect(self.save_image)
        control_layout.addWidget(self.save_button)
        
        self.layout.addWidget(control_frame)
        
        # Görüntü verisi
        self.current_image = None
        self.processed_image = None
        
        # İşlem seçimine göre kontrolleri güncelle
        self.operation_combo.currentIndexChanged.connect(self.update_controls)
        self.update_controls()

    def update_controls(self):
        operation = self.operation_combo.currentText()
        is_sigmoid = operation in ["Standart Sigmoid", "Yatay Kaydırılmış Sigmoid", "Eğimli Sigmoid", "Özel Fonksiyon"]
        self.alpha_spin.setVisible(is_sigmoid)
        self.beta_spin.setVisible(operation in ["Yatay Kaydırılmış Sigmoid", "Eğimli Sigmoid"])
        is_deblur = operation in ["Deblur"]
        self.kernel_size_spin.setVisible(is_deblur)
        self.angle_spin.setVisible(is_deblur)

    def load_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç",
                                                     "", "Image Files (*.png *.jpg *.jpeg)")
            if file_name:
                file_name = str(file_name)
                self.current_image = cv2.imdecode(
                    np.fromfile(file_name, dtype=np.uint8), 
                    cv2.IMREAD_UNCHANGED
                )
                
                if self.current_image is None:
                    raise Exception("Görüntü yüklenemedi")
                    
                self.display_image(self.current_image, self.original_frame.image_label)
                self.processed_image = None
                self.processed_frame.image_label.clear()
                
        except Exception as e:
            print(f"Hata: {str(e)}")
            QMessageBox.critical(self, "Hata", 
                               "Görüntü yüklenirken bir hata oluştu!")

    def display_image(self, image, label):
        if image is not None:
            if len(image.shape) == 2:  # Grayscale image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label.width(), label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)

    def apply_operation(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyin!")
            return

        operation = self.operation_combo.currentText()
        
        try:
            if operation == "Standart Sigmoid":
                self.processed_image = self.apply_standard_sigmoid(self.current_image)
            elif operation == "Yatay Kaydırılmış Sigmoid":
                self.processed_image = self.apply_shifted_sigmoid(self.current_image)
            elif operation == "Eğimli Sigmoid":
                self.processed_image = self.apply_sloped_sigmoid(self.current_image)
            elif operation == "Özel Fonksiyon":
                self.processed_image = self.apply_custom_function(self.current_image)
            elif operation == "Yol Çizgisi Tespiti":
                self.processed_image = self.detect_road_lines(self.current_image)
            elif operation == "Göz Tespiti":
                self.processed_image = self.detect_eyes(self.current_image)
            elif operation == "Koyu Yeşil Bölge Analizi (Excel Çıktısı)":
                self.processed_image = self.analyze_dark_green_regions(self.current_image)
            elif operation == "Deblur":
                kernel_size = self.kernel_size_spin.value()
                angle = self.angle_spin.value()
                self.processed_image = self.advanced_unsharp_deblurring(self.current_image, kernel_size, angle, amount=2.5, color_boost=1.15)
            self.display_image(self.processed_image, self.processed_frame.image_label)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"İşlem sırasında bir hata oluştu: {str(e)}")

    def apply_standard_sigmoid(self, image):
        alpha = self.alpha_spin.value()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        sigmoid = 1 / (1 + np.exp(-alpha * (normalized - 0.5)))
        return (sigmoid * 255).astype(np.uint8)

    def apply_shifted_sigmoid(self, image):
        alpha = self.alpha_spin.value()
        beta = self.beta_spin.value()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        shifted = normalized + beta / 255.0
        sigmoid = 1 / (1 + np.exp(-alpha * (shifted - 0.5)))
        return (sigmoid * 255).astype(np.uint8)

    def apply_sloped_sigmoid(self, image):
        alpha = self.alpha_spin.value()
        beta = self.beta_spin.value()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        sloped = normalized * (1 + beta / 255.0)
        sigmoid = 1 / (1 + np.exp(-alpha * (sloped - 0.5)))
        return (sigmoid * 255).astype(np.uint8)

    def apply_custom_function(self, image):
        alpha = self.alpha_spin.value()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        
        # Özel fonksiyon: Çift sigmoid
        sigmoid1 = 1 / (1 + np.exp(-alpha * (normalized - 0.3)))
        sigmoid2 = 1 / (1 + np.exp(-alpha * (normalized - 0.7)))
        custom = (sigmoid1 + sigmoid2) / 2
        
        return (custom * 255).astype(np.uint8)

    def detect_road_lines(self, image):
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gürültüyü azalt
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Kenar tespiti (eşikleri düşürdük)
        edges = cv2.Canny(blur, 30, 100)

        # Hough Transform ile çizgi tespiti (parametreler güncellendi)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=20,   # Kısa çizgiler için küçük değer
            maxLineGap=40      # Aralıklı çizgiler için büyük değer
        )

        # Sonuç görüntüsünü oluştur
        result = image.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result

    def detect_eyes(self, image):
        try:
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            eye_glasses_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
            if not os.path.exists(eye_cascade_path):
                raise Exception("Haar cascade dosyası bulunamadı!")
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            eye_glasses_cascade = cv2.CascadeClassifier(eye_glasses_cascade_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Tüm görüntüde göz ara
            eyes = eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(120, 120)
            )

            # Eğer göz bulunamazsa gözlüklü göz kaskadını dene
            if len(eyes) == 0:
                eyes = eye_glasses_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    maxSize=(120, 120)
                )

            result = image.copy()
            if len(eyes) == 0:
                QMessageBox.warning(self, "Uyarı", "Göz tespit edilemedi!")
                return image

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(result, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                center_x = ex + ew//2
                center_y = ey + eh//2
                cv2.circle(result, (center_x, center_y), 2, (0, 0, 255), -1)
            return result

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Göz tespiti sırasında bir hata oluştu: {str(e)}")
            return image

    def analyze_dark_green_regions(self, image):
        # 1. Görüntüyü HSV'ye çevir
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 2. Koyu yeşil için maske oluştur (HSV aralığı ayarlanabilir)
        lower = np.array([35, 40, 20])
        upper = np.array([85, 255, 120])
        mask = cv2.inRange(hsv, lower, upper)
        # 3. Bağlı bileşenleri bul
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        results = []
        for i in range(1, num_labels):  # 0 arka plan
            x, y, w, h, area = stats[i]
            if area < 10:  # çok küçük bölgeleri atla
                continue
            region_mask = (labels == i)
            region_pixels = image[region_mask]
            gray_pixels = cv2.cvtColor(region_pixels.reshape(-1,1,3), cv2.COLOR_BGR2GRAY).flatten()
            # Özellikler
            center = tuple(map(int, centroids[i]))
            length, width = w, h
            diagonal = int(np.sqrt(w**2 + h**2))
            energy = np.sum(gray_pixels.astype(np.float32)**2) / 1e4  # ölçekli
            p = np.histogram(gray_pixels, bins=256, range=(0,255))[0]/len(gray_pixels)
            entropy = -np.sum(p * np.log2(p+1e-8))
            mean = int(np.mean(gray_pixels))
            median = int(np.median(gray_pixels))
            results.append([len(results)+1, f"{center[0]},{center[1]}", f"{length} px", f"{width} px", f"{diagonal} px", round(energy,3), round(entropy,2), mean, median])
        # 4. Excel'e yaz
        df = pd.DataFrame(results, columns=["No","Center","Length","Width","Diagonal","Energy","Entropy","Mean","Median"])
        file_name, _ = QFileDialog.getSaveFileName(self, "Excel Olarak Kaydet", "", "Excel Files (*.xlsx)")
        if file_name:
            df.to_excel(file_name, index=False)
        return image  # Görüntüde değişiklik yok, sadece analiz

    def advanced_unsharp_deblurring(self, image, kernel_size=15, angle=0, amount=2.5, color_boost=1.15):
        # Renkli çalış: Her kanal için uygula
        channels = cv2.split(image)
        result_channels = []
        h, w = channels[0].shape

        def motion_blur_psf(length, angle):
            psf = np.zeros((length, length))
            center = length // 2
            rad = np.deg2rad(angle)
            for i in range(length):
                x = int(center + (i - center) * np.cos(rad))
                y = int(center + (i - center) * np.sin(rad))
                if 0 <= x < length and 0 <= y < length:
                    psf[y, x] = 1
            psf /= psf.sum()
            return psf

        psf = motion_blur_psf(kernel_size, angle)
        psf_padded = np.zeros((h, w), dtype=np.float32)
        kh, kw = psf.shape
        y0 = (h - kh) // 2
        x0 = (w - kw) // 2
        psf_padded[y0:y0+kh, x0:x0+kw] = psf

        for ch in channels:
            blurred = cv2.filter2D(ch.astype(np.float32), -1, psf_padded)
            details = ch.astype(np.float32) - blurred
            sharpened = ch.astype(np.float32) + amount * details
            sharpened = np.clip(sharpened, 0, 255)
            result_channels.append(sharpened.astype(np.uint8))

        sharpened_bgr = cv2.merge(result_channels)
        # Renk doygunluğunu artırmak için HSV'ye çevirip S kanalını güçlendir
        hsv = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,1] = np.clip(hsv[...,1] * color_boost, 0, 255)
        hsv[...,2] = np.clip(hsv[...,2], 0, 255)
        colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return colored

    def save_image(self):
        if self.processed_image is not None:
            try:
                file_name, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet",
                                                         "", "PNG Files (*.png);;JPEG Files (*.jpg)")
                if file_name:
                    is_success, im_buf_arr = cv2.imencode(".png", self.processed_image)
                    if is_success:
                        im_buf_arr.tofile(file_name)
                    else:
                        raise Exception("Görüntü kaydedilemedi")
            except Exception as e:
                print(f"Hata: {str(e)}")
                QMessageBox.critical(self, "Hata", 
                                   "Görüntü kaydedilirken bir hata oluştu!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_()) 