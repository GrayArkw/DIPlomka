import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import queue
import logging
from typing import Optional, Dict, Any
from camera_managers import CameraManager
from face_recognition import FaceRecognitionSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Цветовая схема
DARK_BG = "#2C3E50"
LIGHT_BG = "#ECF0F1"
ACCENT = "#3498DB"
SUCCESS = "#2ECC71"
WARNING = "#F39C12"
DANGER = "#E74C3C"
TEXT_LIGHT = "#FFFFFF"
TEXT_DARK = "#2C3E50"


class VideoViewer(tk.Canvas):
    """Виджет для отображения видео с камеры"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(bg=DARK_BG, highlightthickness=0)
        self.image_queue = queue.Queue(maxsize=1)
        self.current_image = None
        self.update_job = None

    def start(self):
        """Запускает обработку очереди кадров"""
        if not self.update_job:
            self._process_queue()

    def stop(self):
        """Останавливает обработку очереди кадров"""
        if self.update_job:
            self.after_cancel(self.update_job)
            self.update_job = None

    def update_frame(self, cv_image):
        """Добавляет новый кадр в очередь"""
        try:
            img = Image.fromarray(cv_image)
            self.image_queue.put_nowait(img)
        except queue.Full:
            pass

    def _process_queue(self):
        """Обрабатывает очередь кадров"""
        try:
            img = self.image_queue.get_nowait()
            self._display_image(img)
        except queue.Empty:
            pass

        self.update_job = self.after(33, self._process_queue)  # ~30 FPS

    def _display_image(self, img: Image.Image):
        """Отображает изображение на холсте"""
        try:
            canvas_width = self.winfo_width()
            canvas_height = self.winfo_height()

            if canvas_width <= 10 or canvas_height <= 10:
                return

            # Масштабирование с сохранением пропорций
            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Создание фона
            bg = Image.new('RGB', (canvas_width, canvas_height), color=DARK_BG)
            offset = ((canvas_width - new_width) // 2, (canvas_height - new_height) // 2)
            bg.paste(img, offset)

            # Обновление холста
            self.current_image = ImageTk.PhotoImage(image=bg)
            self.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.current_image
            )
        except Exception as e:
            logging.error(f"Display image error: {str(e)}")


class AttendanceSystemUI:
    """Графический интерфейс системы учета посещаемости"""

    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x650")
        self.root.minsize(800, 600)
        self.root.configure(bg=DARK_BG)

        # Инициализация системы
        self.available_cameras = CameraManager.get_available_cameras()
        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras found!")
            self.root.destroy()
            return

        self.current_camera_index = self.available_cameras[0]
        self.fr_system = FaceRecognitionSystem(self.current_camera_index)

        # Переменные интерфейса
        self.student_name = tk.StringVar()
        self.student_id = tk.StringVar()
        self.status_message = tk.StringVar(value="System ready")

        # Настройка интерфейса
        self._setup_ui()
        self._start_video_stream()

        # Обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Главный контейнер
        main_frame = tk.Frame(self.root, bg=DARK_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Вкладки
        self.tab_control = ttk.Notebook(main_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)

        # Создание вкладок
        self._create_register_tab()
        self._create_attendance_tab()
        self._create_logs_tab()

        # Нижний колонтитул
        self._create_footer(main_frame)

        # Меню
        self._create_menu()

    def _create_menu(self):
        """Создание меню приложения"""
        menubar = tk.Menu(self.root)

        # Меню выбора камеры
        camera_menu = tk.Menu(menubar, tearoff=0)
        for i in self.available_cameras:
            camera_menu.add_command(
                label=f"Camera {i}",
                command=lambda idx=i: self._switch_camera(idx)
            )
        menubar.add_cascade(label="Select Camera", menu=camera_menu)

        self.root.config(menu=menubar)

    def _create_register_tab(self):
        """Вкладка регистрации студентов"""
        tab = ttk.Frame(self.tab_control)
        self.tab_control.add(tab, text="Register Student")

        # Основной фрейм
        main_frame = tk.Frame(tab, bg=LIGHT_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Форма ввода
        form_frame = tk.Frame(main_frame, bg=LIGHT_BG)
        form_frame.pack(fill=tk.X, pady=10)

        # Поля ввода
        tk.Label(
            form_frame,
            text="Student Full Name:",
            font=("Helvetica", 12),
            bg=LIGHT_BG,
            fg=TEXT_DARK
        ).grid(row=0, column=0, sticky="w", pady=10)

        tk.Entry(
            form_frame,
            textvariable=self.student_name,
            font=("Helvetica", 12),
            bd=2,
            relief="groove"
        ).grid(row=0, column=1, sticky="we", pady=10, padx=10)

        tk.Label(
            form_frame,
            text="Student ID Number:",
            font=("Helvetica", 12),
            bg=LIGHT_BG,
            fg=TEXT_DARK
        ).grid(row=1, column=0, sticky="w", pady=10)

        tk.Entry(
            form_frame,
            textvariable=self.student_id,
            font=("Helvetica", 12),
            bd=2,
            relief="groove"
        ).grid(row=1, column=1, sticky="we", pady=10, padx=10)

        # Кнопки
        btn_frame = tk.Frame(form_frame, bg=LIGHT_BG)
        btn_frame.grid(row=1, column=2, sticky="w")

        tk.Button(
            btn_frame,
            text="Clear Fields",
            font=("Helvetica", 10),
            command=self._clear_registration_fields,
            bg=WARNING,
            fg=TEXT_LIGHT
        ).pack(side=tk.LEFT, padx=10)

        # Инструкции
        tk.Label(
            main_frame,
            text="1. Enter student name and ID\n2. Click Register Student\n3. Follow on-screen instructions",
            font=("Helvetica", 11),
            bg=LIGHT_BG,
            fg=TEXT_DARK,
            justify=tk.LEFT
        ).pack(fill=tk.X, pady=20)

        # Кнопка регистрации
        tk.Button(
            main_frame,
            text="Register Student",
            font=("Helvetica", 12),
            command=self._start_registration,
            bg=ACCENT,
            fg=TEXT_LIGHT
        ).pack(pady=20)

    def _create_attendance_tab(self):
        """Вкладка учета посещаемости"""
        tab = ttk.Frame(self.tab_control)
        self.tab_control.add(tab, text="Take Attendance")

        # Основной фрейм
        main_frame = tk.Frame(tab, bg=LIGHT_BG)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Информация
        tk.Label(
            main_frame,
            text="Click 'Start Scanning' to take attendance",
            font=("Helvetica", 12),
            bg=LIGHT_BG,
            fg=TEXT_DARK
        ).pack(pady=20)

        # Видео контейнер
        video_container = tk.Frame(main_frame, bg=DARK_BG)
        video_container.pack(fill=tk.BOTH, expand=True)

        # Видео просмотр
        self.video_viewer = VideoViewer(video_container)
        self.video_viewer.pack(fill=tk.BOTH, expand=True)

        # Кнопка сканирования
        btn_frame = tk.Frame(main_frame, bg=LIGHT_BG)
        btn_frame.pack(fill=tk.X, pady=20)

        self.scan_btn = tk.Button(
            btn_frame,
            text="Start Scanning",
            font=("Helvetica", 12),
            command=self._toggle_scanning,
            bg=ACCENT,
            fg=TEXT_LIGHT
        )
        self.scan_btn.pack()

    def _create_logs_tab(self):
        """Вкладка журнала посещаемости"""
        tab = ttk.Frame(self.tab_control)
        self.tab_control.add(tab, text="Attendance Logs")

        # Основной фрейм
        main_frame = tk.Frame(tab, bg=LIGHT_BG)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        tk.Label(
            main_frame,
            text="Attendance Records",
            font=("Helvetica", 16),
            bg=LIGHT_BG,
            fg=TEXT_DARK
        ).pack(pady=20)

        # Таблица
        tree_frame = tk.Frame(main_frame, bg=LIGHT_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Скроллбар
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Настройка стиля
        style = ttk.Style()
        style.configure("Treeview",
                        background=LIGHT_BG,
                        foreground=TEXT_DARK,
                        rowheight=25,
                        fieldbackground=LIGHT_BG)
        style.map('Treeview', background=[('selected', ACCENT)])
        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))

        # Таблица
        self.attendance_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "name", "date", "time"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.attendance_tree.yview)

        # Настройка колонок
        self.attendance_tree.heading("id", text="Student ID")
        self.attendance_tree.heading("name", text="Student Name")
        self.attendance_tree.heading("date", text="Date")
        self.attendance_tree.heading("time", text="Time")

        self.attendance_tree.column("id", width=100, anchor="center")
        self.attendance_tree.column("name", width=250, anchor="w")
        self.attendance_tree.column("date", width=150, anchor="center")
        self.attendance_tree.column("time", width=150, anchor="center")

        self.attendance_tree.pack(fill=tk.BOTH, expand=True)

        # Кнопка обновления
        btn_frame = tk.Frame(main_frame, bg=LIGHT_BG)
        btn_frame.pack(fill=tk.X, padx=20, pady=20)

        tk.Button(
            btn_frame,
            text="Refresh Records",
            font=("Helvetica", 11),
            command=self._load_attendance_data,
            bg=ACCENT,
            fg=TEXT_LIGHT
        ).pack(side=tk.LEFT, padx=10)

        # Загрузка данных
        self._load_attendance_data()

    def _create_footer(self, parent):
        """Создание нижней панели"""
        footer = tk.Frame(parent, bg=DARK_BG)
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        # Статус
        tk.Label(
            footer,
            textvariable=self.status_message,
            font=("Helvetica", 10),
            fg=TEXT_LIGHT,
            bg=DARK_BG
        ).pack(side=tk.LEFT, padx=20)

        # Версия
        tk.Label(
            footer,
            text="v2.2.0",
            font=("Helvetica", 10),
            fg=TEXT_LIGHT,
            bg=DARK_BG
        ).pack(side=tk.RIGHT, padx=20)

    def _start_video_stream(self):
        """Запуск потока обновления видео"""
        self.video_viewer.start()

        def update():
            ret, frame = self.fr_system.camera_manager.get_frame()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_viewer.update_frame(frame)
            self.root.after(50, update)  # ~20 FPS

        update()

    def _switch_camera(self, new_index: int):
        """Переключение на другую камеру"""
        self.current_camera_index = new_index
        self.fr_system.camera_manager.switch_camera(new_index)
        messagebox.showinfo("Info", f"Switched to Camera {new_index}")

    def _clear_registration_fields(self):
        """Очистка полей регистрации"""
        self.student_name.set("")
        self.student_id.set("")
        self.status_message.set("Fields cleared")

    def _validate_inputs(self) -> bool:
        """Проверка введенных данных"""
        name = self.student_name.get().strip()
        student_id = self.student_id.get().strip()

        if not name or not name.replace(" ", "").isalpha():
            messagebox.showerror("Error", "Please enter a valid name (letters only)")
            return False

        if not student_id or not student_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid numeric ID")
            return False

        if self.fr_system.student_id_exists(student_id):
            messagebox.showerror("Error", "Student with this ID already exists!")
            return False

        return True

    def _start_registration(self):
        """Начало процесса регистрации"""
        if not self._validate_inputs():
            return

        name = self.student_name.get().strip()
        student_id = self.student_id.get().strip()

        self.status_message.set("Starting registration...")

        # Запуск в отдельном потоке
        threading.Thread(
            target=self._process_registration,
            args=(name, student_id),
            daemon=True
        ).start()

    def _process_registration(self, name: str, student_id: str):
        """Обработка регистрации студента"""

        def progress_callback(**kwargs):
            if 'instruction' in kwargs:
                self.status_message.set(
                    f"Instruction: {kwargs['instruction']} "
                    f"(Remaining: {kwargs.get('remaining', 0)})"
                )

            if 'frame' in kwargs:
                frame = kwargs['frame']
                if 'face_box' in kwargs:
                    x, y, w, h = kwargs['face_box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_viewer.update_frame(frame)

            if 'count' in kwargs:
                self.status_message.set(f"Captured {kwargs['count']} images")

        success, count = self.fr_system.take_images(name, student_id, progress_callback)

        if success:
            messagebox.showinfo(
                "Success",
                f"Successfully registered {name} with {count} images"
            )
            self._clear_registration_fields()
            self.status_message.set("Training model...")
            self.fr_system.train_model()
            self.status_message.set("Registration complete")
        else:
            messagebox.showerror(
                "Error",
                f"Failed to register student. Captured {count} images"
            )
            self.status_message.set("Registration failed")

    def _toggle_scanning(self):
        """Переключение режима сканирования"""
        if self.fr_system.is_scanning:
            self.fr_system.is_scanning = False
            self.scan_btn.config(text="Start Scanning", bg=ACCENT)
            self.status_message.set("Scanning stopped")
        else:
            self.fr_system.is_scanning = True
            self.scan_btn.config(text="Stop Scanning", bg=DANGER)
            self.status_message.set("Starting scanning...")

            # Запуск в отдельном потоке
            threading.Thread(
                target=self._process_scanning,
                daemon=True
            ).start()

    def _process_scanning(self):
        """Обработка сканирования для учета посещаемости"""

        def update_callback(**kwargs):
            if 'frame' in kwargs:
                frame = cv2.cvtColor(kwargs['frame'], cv2.COLOR_BGR2RGB)
                self.video_viewer.update_frame(frame)

            if 'status' in kwargs:
                self.status_message.set(kwargs['status'])

            return self.fr_system.is_scanning

        success, message = self.fr_system.track_attendance(update_callback)

        if not success:
            messagebox.showerror("Error", message)

        self.fr_system.is_scanning = False
        self.scan_btn.config(text="Start Scanning", bg=ACCENT)
        self._load_attendance_data()
        self.status_message.set(message)

    def _load_attendance_data(self):
        """Загрузка данных о посещаемости в таблицу"""
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        records = self.fr_system.load_attendance_data()
        for record in records:
            self.attendance_tree.insert(
                "",
                "end",
                values=(
                    record.get("ID", ""),
                    record.get("NAME", ""),
                    record.get("DATE", ""),
                    record.get("TIME", "")
                )
            )

        self.status_message.set(f"Loaded {len(records)} attendance records")

    def _on_closing(self):
        """Обработчик закрытия окна"""
        self.fr_system.is_scanning = False
        self.video_viewer.stop()
        self.fr_system.camera_manager.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemUI(root)
    root.mainloop()