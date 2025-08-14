# Face Recognition Attendance System with UI 👤💻

A complete attendance tracking solution with graphical interface that handles all operations in one application.

## Features ✨
- **All-in-One Interface** - No need to run separate scripts
- **User Management** - Add/register new users directly
- **Automatic Training** - Model trains after capturing faces
- **Real-Time Recognition** - Live attendance recording
- **Data Export** - Save records to CSV with one click
How to Use 🚀
1. Launch the Application
bash
python Face_Recognition.py
2. Main Interface Components
Left Panel: Live camera feed

Right Panel: Controls and attendance records

3. Complete Workflow
Register New User:

Enter User ID and Name

Click "Capture Dataset"

Face the camera (system captures 30 samples automatically)

Train Model:

Click "Train Model" (automatic after capturing)

Wait for training confirmation

Take Attendance:

Click "Start Recognition"

System will:

Detect faces in real-time

Identify registered users

Record attendance automatically

Export Data:

Click "Export Attendance"

CSV file saved as attendance.csv

File Structure
text
project/
├── Face_Recognition.py       # Main application (with UI)
├── haarcascade_frontalface_default.xml
├── dataset/                  # Auto-created for face samples
├── trainer.yml               # Auto-created during training
└── attendance.csv            # Auto-created when exporting
Troubleshooting ⚠️
Common Issues	Solutions
Camera not detected	Check camera permissions/connections
Low recognition accuracy	Ensure good lighting and retrain model
Module errors	Reinstall requirements: pip install -r requirements.txt
Requirements
Python 3.6+

Webcam

Windows/macOS/Linux

Key advantages of this version:
1. **Single-File Operation** - Everything runs through `Face_Recognition.py`
2. **Visual Workflow** - Clear diagram of the process
3. **Troubleshooting Table** - Quick solutions to common problems
4. **Simplified Instructions** - Focused on the UI workflow
5. **Auto-created Files** - Highlights which files are generated automatically

An intuitive desktop application that:

Registers Employees/Students:

Capture face samples through webcam

Assign unique IDs and names

Automates Attendance:

Recognizes faces in real-time

Prevents duplicate entries for same day

Manages Records:

View attendance history in-app

Export data to spreadsheets

Works offline after initial setup

Key Technical Specifications
Component	Technology Used	Purpose
Face Detection	Haar Cascades (OpenCV)	Locate faces in video feed
Face Recognition	LBPH Algorithm	Identify registered users
User Interface	Tkinter (Custom Styled)	Interactive controls
Data Storage	CSV Files	Attendance records



<img width="528" height="1154" alt="Face_Recognition" src="https://github.com/user-attachments/assets/60c60024-a3fd-41e8-aaea-217b41e21117" />


Features Breakdown
User Registration:

Validates unique user IDs

Auto-captures 30 face variations

Progress indicators during capture

Attendance Mode:

Real-time face bounding boxes

Confidence level display

Timestamped entries

Admin Controls:

Reset functionality

Training status monitoring

Export triggers

Ideal Use Cases
Corporate offices for employee attendance

Schools/Universities for student tracking

Events for registered guest check-ins

Secure facilities for authorized access 
