import os
import csv
import time
import datetime
import numpy as np
import pandas as pd
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Callable
from PIL import Image
from camera_managers import CameraManager

logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """Система распознавания лиц для учета посещаемости"""

    def __init__(self, camera_index: int = 0):
        self.camera_manager = CameraManager(camera_index)
        self.is_scanning = False
        self._setup_directories()

    def _setup_directories(self):
        """Создание необходимых директорий"""
        os.makedirs("TrainingImages", exist_ok=True)
        os.makedirs("ImagesUnknown", exist_ok=True)

        if not os.path.exists("StudentDetails.csv"):
            with open('StudentDetails.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "NAME"])

        if not os.path.exists("Attendance.csv"):
            with open('Attendance.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "NAME", "DATE", "TIME"])

    def student_id_exists(self, student_id: str) -> bool:
        """Проверяет, существует ли студент с заданным ID"""
        if not os.path.exists("StudentDetails.csv"):
            return False

        with open('StudentDetails.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            return any(row and row[0] == student_id for row in reader)

    def take_images(
            self,
            name: str,
            student_id: str,
            progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, int]:
        """
        Захват изображений для регистрации студента
        """
        try:
            detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            sample_num = 0
            instructions = [
                "Look straight with neutral face",
                "Smile slightly",
                "Turn head left",
                "Turn head right",
                "Look up slightly"
            ]

            for instruction in instructions:
                if progress_callback:
                    progress_callback(
                        instruction=instruction,
                        remaining=len(instructions) - instructions.index(instruction) - 1
                    )

                for _ in range(10):
                    ret, frame = self.camera_manager.get_frame()
                    if not ret:
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

                    if len(faces) == 0:
                        continue

                    for (x, y, w, h) in faces:
                        sample_num += 1
                        img_path = f"TrainingImages/{name}.{student_id}.{sample_num}.jpg"
                        cv2.imwrite(img_path, gray[y:y + h, x:x + w])

                        if progress_callback:
                            progress_callback(
                                frame=frame,
                                count=sample_num,
                                face_box=(x, y, w, h)
                            )

            with open('StudentDetails.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([student_id, name])

            logger.info(f"Registered student {name} with {sample_num} images")
            return True, sample_num

        except Exception as e:
            logger.error(f"Failed to register student: {str(e)}")
            return False, 0

    def train_model(self) -> Tuple[bool, str]:
        """Обучение модели распознавания лиц"""
        try:
            if not os.path.exists("TrainingImages") or not os.listdir("TrainingImages"):
                return False, "No training images found"

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces, ids = self._get_images_and_labels("TrainingImages")

            if len(faces) == 0:
                return False, "No faces detected in training images"

            recognizer.train(faces, np.array(ids))
            recognizer.save("Trainer.yml")
            logger.info("Model trained successfully")
            return True, "Model training completed"

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False, f"Training error: {str(e)}"

    def _get_images_and_labels(self, path: str) -> Tuple[List[np.ndarray], List[int]]:
        """Получение изображений и меток из указанной директории"""
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        ids = []

        for image_path in image_paths:
            try:
                img = Image.open(image_path).convert('L')
                img_np = np.array(img, 'uint8')
                id_num = int(os.path.split(image_path)[-1].split(".")[1])
                faces.append(img_np)
                ids.append(id_num)
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {str(e)}")
                continue

        return faces, ids

    def track_attendance(
            self,
            update_callback: Optional[Callable] = None,
            confidence_threshold: float = 60.0
    ) -> Tuple[bool, str]:
        """
        Отслеживание посещаемости в реальном времени
        """
        try:
            if not os.path.exists("Trainer.yml"):
                return False, "Model not trained"

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("Trainer.yml")

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            try:
                df = pd.read_csv("StudentDetails.csv")
            except:
                df = pd.DataFrame(columns=["ID", "NAME"])

            font = cv2.FONT_HERSHEY_SIMPLEX
            recognized_ids = set()

            while self.is_scanning:
                ret, frame = self.camera_manager.get_frame()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    id_num, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                    if confidence < confidence_threshold:
                        student_name = df.loc[df['ID'] == id_num, 'NAME'].values[0] \
                            if id_num in df['ID'].values else "Unknown"

                        if id_num not in recognized_ids:
                            self._record_attendance(id_num, student_name)
                            recognized_ids.add(id_num)
                            if update_callback:
                                update_callback(
                                    status=f"Recognized: {student_name} (ID: {id_num})",
                                    recognized=True
                                )

                        label = f"{student_name} ({confidence:.1f}%)"
                        color = (0, 255, 0)
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)
                        self._save_unknown_face(gray[y:y + h, x:x + w])

                    cv2.putText(
                        frame, label, (x, y + h + 30),
                        font, 0.8, color, 2
                    )

                if update_callback and not update_callback(frame=frame):
                    break

            return True, "Attendance tracking completed"

        except Exception as e:
            logger.error(f"Attendance tracking failed: {str(e)}")
            return False, f"Error: {str(e)}"

    def _record_attendance(self, student_id: str, name: str):
        """Запись посещаемости в CSV"""
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        with open('Attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([student_id, name, date, time_str])
        logger.info(f"Attendance recorded for {name} (ID: {student_id})")

    def _save_unknown_face(self, face_img: np.ndarray):
        """Сохранение изображения неизвестного лица"""
        unknown_count = len(os.listdir("ImagesUnknown"))
        cv2.imwrite(f"ImagesUnknown/Unknown_{unknown_count + 1}.jpg", face_img)
        logger.debug("Saved unknown face image")

    def load_attendance_data(self) -> List[Dict[str, str]]:
        """Загрузка данных о посещаемости"""
        try:
            if not os.path.exists("Attendance.csv"):
                return []

            df = pd.read_csv("Attendance.csv")
            if df.empty:
                return []

            df['DATETIME'] = pd.to_datetime(
                df['DATE'] + ' ' + df['TIME'],
                format='%d-%m-%Y %H:%M:%S',
                errors='coerce'
            )
            return df.sort_values('DATETIME', ascending=False) \
                .dropna() \
                .to_dict('records')

        except Exception as e:
            logger.error(f"Failed to load attendance data: {str(e)}")
            return []