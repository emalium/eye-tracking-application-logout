import os
from multiprocessing import current_process
import atexit
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import dlib
import base64
import time
from math import hypot

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a random secret key
socketio = SocketIO(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Dummy user class and database for demonstration
class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {'user@example.com': {'password': 'password'}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variables for blink detection
blink_count = 0
last_blink_time = 0
double_blink_threshold = 3  # seconds

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email]['password'] == password:
            login_user(User(email))
            return redirect(url_for('dashboard'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/eye-tracking')
@login_required
def eye_tracking():
    return render_template('eye_tracking.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Eye-tracking related functions

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def process_frame(frame):
    global blink_count, last_blink_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            current_time = time.time()
            if current_time - last_blink_time < double_blink_threshold:
                blink_count += 1
                if blink_count == 2:
                    return "double_blink"
            else:
                blink_count = 1
            last_blink_time = current_time
            return "blink"

    return "no_blink"

@socketio.on("video_frame")
def handle_video_frame(data):
    img_data = base64.b64decode(data.split(",")[1])
    img_np = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(img_np, flags=1)

    result = process_frame(frame)
    if result == "blink":
        emit("blink_detected", {"message": "Blink detected!"})
    elif result == "double_blink":
        emit("logout", {"message": "Double blink detected! Logging out..."})
        logout_user()

def cleanup_resources():
    # Add any cleanup code here if needed
    pass

atexit.register(cleanup_resources)

if __name__ == "__main__":
    if current_process().name == 'MainProcess':
        os.environ['WERKZEUG_RUN_MAIN'] = 'true'
    socketio.run(app, debug=True, use_reloader=False)