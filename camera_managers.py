import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class CameraManager:
    """Менеджер для работы с локальными камерами"""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    @staticmethod
    def get_available_cameras() -> List[int]:
        """Возвращает список доступных камер"""
        available = []
        for i in range(3):  # Проверяем первые 3 индекса
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def start_camera(self):
        """Инициализирует камеру"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_index}")

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Получает кадр с камеры"""
        if self.cap is None:
            self.start_camera()
        return self.cap.read()

    def release(self):
        """Освобождает ресурсы камеры"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def switch_camera(self, new_index: int):
        """Переключает на другую камеру"""
        self.release()
        self.camera_index = new_index
        self.start_camera()