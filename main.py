"""
Face Recognition Attendance Management System
Main Application Entry Point
"""

import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import os
import pandas as pd
from datetime import datetime
import threading

from face_recognition_attendance.face_recognition_system import FaceRecognitionSystem
from face_recognition_attendance.attendance_manager   import AttendanceManager
from face_recognition_attendance.ui_components       import (
    RegisterStudentWindow,
    AttendanceViewerWindow,
    ManualAttendanceWindow,
    SettingsWindow
)

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        # Initialize face system and attendance manager
        self.face_system = FaceRecognitionSystem()
        self.attendance_manager = AttendanceManager()

        # Create directories
        self._ensure_directories()

        # Setup UI
        self._setup_ui()

    def _ensure_directories(self):
        paths = [
            'data/student_images',
            'data/attendance',
            'data/trained_model',
            'assets/icons'
        ]
        for p in paths:
            os.makedirs(p, exist_ok=True)

    def _setup_ui(self):
        title = tk.Label(self.root,
                         text="Face Recognition Attendance System",
                         font=('Arial', 24, 'bold'),
                         bg='#2c3e50', fg='#ecf0f1')
        title.pack(pady=20)

        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        left = tk.Frame(main_frame, bg='#34495e', bd=2, relief='raised')
        left.pack(side='left', fill='y', padx=(0,10))
        self._create_buttons(left)

        right = tk.Frame(main_frame, bg='#34495e', bd=2, relief='raised')
        right.pack(side='right', fill='both', expand=True)

        self.video_label = tk.Label(right, bg='black', text="Camera Feed",
                                    fg='white', font=('Arial',16))
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)

        status_frame = tk.Frame(right, bg='#34495e')
        status_frame.pack(fill='x', padx=10, pady=(0,5))
        self.status_label = tk.Label(status_frame, text="Status: Ready",
                                     bg='#34495e', fg='#2ecc71',
                                     font=('Arial',12))
        self.status_label.pack()

        info_frame = tk.Frame(right, bg='#34495e')
        info_frame.pack(fill='both', padx=10, pady=(0,10))
        self.info_text = tk.Text(info_frame, height=8,
                                 bg='#2c3e50', fg='#ecf0f1',
                                 font=('Courier',10))
        self.info_text.pack(fill='both', expand=True)

    def _create_buttons(self, parent):
        btns = [
            ("Register New Student", self._open_register),
            ("Train Model", self._train_model),
            ("Start Attendance", self._start_attendance),
            ("Stop Attendance", self._stop_attendance),
            ("View Attendance", self._view_attendance),
            ("Manual Override", self._manual_override),
            ("Settings", self._open_settings),
            ("Exit", self._exit_app),
        ]
        for text, cmd in btns:
            b = tk.Button(parent, text=text, command=cmd,
                          font=('Arial',12,'bold'),
                          bg='#3498db', fg='white', relief='flat',
                          cursor='hand2', height=2)
            b.pack(fill='x', pady=5)

    def _open_register(self):
        RegisterStudentWindow(self.root, self.face_system)

    def _train_model(self):
        self.status_label.config(text="Status: Training...", fg='#f39c12')
        def worker():
            ok = self.face_system.train_model()
            if ok:
                self.status_label.config(text="Status: Trained", fg='#2ecc71')
                messagebox.showinfo("Success", "Model trained successfully")
            else:
                self.status_label.config(text="Status: Training Failed", fg='#e74c3c')
                messagebox.showerror("Error", "Model training failed")
        threading.Thread(target=worker, daemon=True).start()

    def _start_attendance(self):
        self.status_label.config(text="Status: Taking Attendance...", fg='#f39c12')
        threading.Thread(target=self._attendance_loop, daemon=True).start()

    def _attendance_loop(self):
        self.face_system.start_attendance_mode(
            self.video_label, self._on_recognized
        )

    def _on_recognized(self, name, conf):
        self.attendance_manager.mark_attendance(name, conf)
        self._update_info()

    def _update_info(self):
        df = self.attendance_manager.get_today_attendance()
        text = f"Today's Attendance ({datetime.now().date()}):\n" + "-"*40 + "\n"
        if df.empty:
            text += "No records yet\n"
        else:
            for _, r in df.iterrows():
                text += f"{r['Name']} at {r['Time']} (Conf: {r['Confidence']:.1f})\n"
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)

    def _stop_attendance(self):
        self.face_system.stop_camera()
        self.status_label.config(text="Status: Stopped", fg='#e74c3c')

    def _view_attendance(self):
        AttendanceViewerWindow(self.root, self.attendance_manager)

    def _manual_override(self):
        ManualAttendanceWindow(self.root, self.attendance_manager)

    def _open_settings(self):
        SettingsWindow(self.root)

    def _exit_app(self):
        if messagebox.askyesno("Exit", "Quit application?"):
            self.face_system.stop_camera()
            self.root.destroy()

def main():
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
