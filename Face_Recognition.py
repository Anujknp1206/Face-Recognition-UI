import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import datetime
import csv

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f2f5")
        
        # Custom colors
        self.primary_color = "#3498db"
        self.secondary_color = "#2980b9"
        self.accent_color = "#e74c3c"
        self.light_color = "#ecf0f1"
        self.dark_color = "#2c3e50"
        
        # Initialize variables
        self.face_id = tk.StringVar()
        self.user_name = tk.StringVar()
        self.attendance_date = tk.StringVar()
        self.current_user = None
        self.capture = None
        self.recognizer = None
        self.face_detector = None
        self.is_training = False
        self.is_recording = False
        self.frame_count = 0
        self.attendance_data = []
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_widgets()
        
        # Load attendance data
        self.load_attendance()
        
    def load_models(self):
        """Load face recognition models"""
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            if os.path.exists('trainer.yml'):
                self.recognizer.read('trainer.yml')
            
            self.face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            if self.face_detector.empty():
                raise Exception("Haar cascade file not loaded properly")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def create_widgets(self):
        """Create all UI widgets"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.light_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera and controls
        left_frame = tk.Frame(main_frame, bg=self.light_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera frame
        self.camera_frame = tk.Label(left_frame, bg="black")
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons frame
        control_frame = tk.Frame(left_frame, bg=self.light_color)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.capture_btn = ttk.Button(control_frame, text="Capture Dataset", command=self.capture_dataset, 
                                    style="Accent.TButton")
        self.capture_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.train_btn = ttk.Button(control_frame, text="Train Model", command=self.train_model,
                                  style="Accent.TButton")
        self.train_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.recognize_btn = ttk.Button(control_frame, text="Start Recognition", command=self.toggle_recognition,
                                      style="Accent.TButton")
        self.recognize_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Right panel - User info and attendance
        right_frame = tk.Frame(main_frame, bg=self.light_color, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # User info frame
        user_frame = tk.LabelFrame(right_frame, text="User Information", bg=self.light_color, 
                                 fg=self.dark_color, font=("Arial", 12, "bold"))
        user_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(user_frame, text="User ID:", bg=self.light_color).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.user_id_entry = ttk.Entry(user_frame, textvariable=self.face_id)
        self.user_id_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        tk.Label(user_frame, text="User Name:", bg=self.light_color).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(user_frame, textvariable=self.user_name).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        tk.Label(user_frame, text="Date:", bg=self.light_color).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(user_frame, textvariable=self.attendance_date).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.attendance_date.set(datetime.date.today().strftime("%Y-%m-%d"))
        
        # Attendance frame
        attendance_frame = tk.LabelFrame(right_frame, text="Attendance Records", bg=self.light_color,
                                        fg=self.dark_color, font=("Arial", 12, "bold"))
        attendance_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Treeview for attendance records
        self.attendance_tree = ttk.Treeview(attendance_frame, columns=("date", "user_id", "user_name", "time"), 
                                           show="headings", selectmode="browse")
        
        self.attendance_tree.heading("date", text="Date")
        self.attendance_tree.heading("user_id", text="User ID")
        self.attendance_tree.heading("user_name", text="Name")
        self.attendance_tree.heading("time", text="Time")
        
        self.attendance_tree.column("date", width=80)
        self.attendance_tree.column("user_id", width=60)
        self.attendance_tree.column("user_name", width=100)
        self.attendance_tree.column("time", width=60)
        
        scrollbar = ttk.Scrollbar(attendance_frame, orient="vertical", command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export button
        export_frame = tk.Frame(right_frame, bg=self.light_color)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Export Attendance", command=self.export_attendance,
                  style="Accent.TButton").pack(fill=tk.X, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        # Configure styles
        self.configure_styles()
        
        # Start camera
        self.start_camera()
    
    def configure_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure button styles
        style.configure("TButton", font=("Arial", 10), padding=5)
        style.configure("Accent.TButton", background=self.primary_color, foreground="white")
        style.map("Accent.TButton",
                 background=[("active", self.secondary_color), ("disabled", "#cccccc")])
        
        # Configure treeview styles
        style.configure("Treeview", font=("Arial", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.map("Treeview", background=[("selected", self.primary_color)])
        
        # Configure label frame styles
        style.configure("TLabelframe", background=self.light_color)
        style.configure("TLabelframe.Label", background=self.light_color, foreground=self.dark_color)
    
    def start_camera(self):
        """Initialize the camera"""
        try:
            self.capture = cv2.VideoCapture(0)
            self.capture.set(3, 640)
            self.capture.set(4, 480)
            self.show_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize camera: {str(e)}")
    
    def show_frame(self):
        """Show camera feed in the UI"""
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                if self.is_recording:
                    frame = self.process_recognition(frame)
                elif self.is_training:
                    frame = self.process_training(frame)
                
                # Convert to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)
                
                self.camera_frame.img = img  # Keep reference to prevent garbage collection
                self.camera_frame.configure(image=img)
            
            self.root.after(10, self.show_frame)
    
    def process_recognition(self, frame):
        """Process frame for face recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if self.recognizer:
                id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                
                if confidence < 100:
                    user_id = str(id)
                    confidence_text = f"{round(100 - confidence)}%"
                    self.current_user = user_id
                    
                    # Add to attendance if not already recorded today
                    self.record_attendance(user_id, f"User {user_id}")
                    
                    # Display user info
                    cv2.putText(frame, f"User {user_id}", (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (x+5, y+h-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                else:
                    cv2.putText(frame, "Unknown", (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def process_training(self, frame):
        """Process frame for training data collection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # Save the face image
            if self.frame_count < 30 and self.frame_count % 5 == 0:  # Capture every 5 frames
                face_id = self.face_id.get()
                if not face_id:
                    messagebox.showwarning("Warning", "Please enter a user ID first")
                    self.is_training = False
                    self.capture_btn.config(text="Capture Dataset")
                    return frame
                
                # Ensure dataset directory exists
                dataset_path = "dataset"
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                
                # Save face image
                cv2.imwrite(f"{dataset_path}/User.{face_id}.{self.frame_count}.jpg", gray[y:y+h, x:x+w])
            
            self.frame_count += 1
            
            # Display countdown
            cv2.putText(frame, f"Capturing: {30 - self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if self.frame_count >= 30:
                self.is_training = False
                self.capture_btn.config(text="Capture Dataset")
                messagebox.showinfo("Info", "Dataset captured successfully!")
                self.frame_count = 0
                break
        
        return frame
    
    def capture_dataset(self):
        """Toggle dataset capture mode"""
        if not self.is_training:
            user_id = self.face_id.get()
            if not user_id:
                messagebox.showwarning("Warning", "Please enter a user ID first")
                return
            
            self.is_training = True
            self.frame_count = 0
            self.capture_btn.config(text="Stop Capturing")
            self.status_var.set("Capturing dataset... Please face the camera")
        else:
            self.is_training = False
            self.capture_btn.config(text="Capture Dataset")
            self.status_var.set("Ready")
    
    def train_model(self):
        """Train the face recognition model"""
        try:
            # Get the path to the dataset
            dataset_path = 'dataset'
            if not os.path.exists(dataset_path):
                messagebox.showerror("Error", "Dataset directory not found!")
                return
            
            # Initialize lists for faces and IDs
            faces = []
            ids = []
            
            # Get the image paths
            image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
            
            if not image_paths:
                messagebox.showerror("Error", "No images found in the dataset directory!")
                return
            
            # Loop through each image
            for image_path in image_paths:
                # Read the image and convert to grayscale
                pil_img = Image.open(image_path).convert('L')
                img_numpy = np.array(pil_img, 'uint8')
                
                # Extract the user ID from the image file name
                user_id = int(os.path.split(image_path)[-1].split(".")[1])
                
                # Detect faces in the image
                detected_faces = self.face_detector.detectMultiScale(img_numpy)
                
                for (x, y, w, h) in detected_faces:
                    faces.append(img_numpy[y:y+h, x:x+w])
                    ids.append(user_id)
            
            # Train the model
            self.recognizer.train(faces, np.array(ids))
            
            # Save the model
            self.recognizer.write('trainer.yml')
            
            messagebox.showinfo("Success", "Model trained successfully!")
            self.status_var.set("Model trained successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Training failed")
    
    def toggle_recognition(self):
        """Toggle face recognition mode"""
        if not self.is_recording:
            if not os.path.exists('trainer.yml'):
                messagebox.showwarning("Warning", "Please train the model first!")
                return
            
            self.is_recording = True
            self.recognize_btn.config(text="Stop Recognition")
            self.status_var.set("Recognition active - Looking for faces...")
        else:
            self.is_recording = False
            self.recognize_btn.config(text="Start Recognition")
            self.status_var.set("Ready")
    
    def record_attendance(self, user_id, user_name):
        """Record attendance for recognized user"""
        today = datetime.date.today().strftime("%Y-%m-%d")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Check if already recorded today
        for record in self.attendance_data:
            if record[0] == today and record[1] == user_id:
                return
        
        # Add new record
        record = (today, user_id, user_name, current_time)
        self.attendance_data.append(record)
        
        # Update treeview
        self.attendance_tree.insert("", tk.END, values=record)
        
        # Update status
        self.status_var.set(f"Attendance recorded for {user_name}")
    
    def load_attendance(self):
        """Load attendance data from file"""
        if os.path.exists('attendance.csv'):
            with open('attendance.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 4:
                        self.attendance_data.append(tuple(row))
                        self.attendance_tree.insert("", tk.END, values=row)
    
    def export_attendance(self):
        """Export attendance data to CSV file"""
        try:
            with open('attendance.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "User ID", "Name", "Time"])
                writer.writerows(self.attendance_data)
            
            messagebox.showinfo("Success", "Attendance data exported to attendance.csv")
            self.status_var.set("Attendance data exported")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export attendance: {str(e)}")
            self.status_var.set("Export failed")
    
    def on_closing(self):
        """Handle window closing event"""
        if self.capture:
            self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()