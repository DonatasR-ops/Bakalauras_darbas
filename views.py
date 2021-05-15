from flask import Blueprint, render_template, Response, send_file, redirect, url_for, request, Flask, flash
from flask_login import login_required, current_user
from gaze_tracker import get_gaze_ratio
import glob
import os
import cv2
import dlib
import time
import csv
from datetime import datetime
import numpy as np
from _init_ import *
from gaze_tracker import *
from werkzeug.utils import secure_filename


views = Blueprint('views', __name__)

camera = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_ITALIC

gaze_left = 0.7
gaze_right = 2


@views.route('/')
def home():
    return render_template("home.html", user=current_user)


@views.route('/gaze-real', methods=['GET', 'POST'])
def gaze_real():
    return render_template("gaze_real.html", user=current_user)


@views.route('/download')
def download_file():
    list_of_files = glob.glob('*.csv')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    p = latest_file
    return send_file(p, as_attachment=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_frames_algorithm():  # generate frame by frame from camera
    time_right = 0
    time_center = 0
    time_left = 0
    while True:
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)
            landmarks = predictor(gray, face)
            gaze_ratio_left_eye = get_gaze_ratio(frame, gray, [36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio(frame, gray, [42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            new_frame = np.zeros((500, 500, 3), np.uint8)
            start = time.time()
            if gaze_ratio >= gaze_right:
                cv2.putText(frame, "LEFT", (x - 100, y), font, 1, (0, 0, 255), 2)
                new_frame[:] = (0, 0, 255)
                end1 = time.time()
                time_left += (end1 - start) * 100
            elif gaze_right > gaze_ratio > gaze_left:
                cv2.putText(frame, "CENTER", (x - 100, y), font, 1, (0, 0, 255), 2)
                end2 = time.time()
                time_center += (end2 - start) * 1000
            else:
                new_frame[:] = (255, 0, 0)
                cv2.putText(frame, "RIGHT", (x - 100, y), font, 1, (0, 0, 255), 2)
                end3 = time.time()
                time_right += (end3 - start) * 100
            print([time_right, time_center, time_left])
            now = datetime.now()
            d3 = dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            with open(d3 + '.csv', 'w', newline='') as f:
                thewriter = csv.writer(f)
                thewriter.writerow(['Gaze direction:', 'Time (s):'])
                thewriter.writerow(['Left', time_left])
                thewriter.writerow(['Right', time_right])
                thewriter.writerow(['Center', time_center])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@views.route('/video_feed_base')
def video_feed_base():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@views.route('/video_feed')
def video_feed():
    return Response(gen_frames_algorithm(), mimetype='multipart/x-mixed-replace; boundary=frame')


@views.route("/gaze-analyse", methods=["GET", "POST"])
def gaze_analyse():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Wrong file format', "error")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            flash('Analysing...')
            filename = secure_filename(file.filename)
            if filename.endswith('.mp4'):
                if analyse_video(os.path.abspath(filename)) == 'Low':
                    flash('Student cheating risk: ' + analyse_video(os.path.abspath(filename)))
                else:
                    flash('Student cheating risk: ' + analyse_video(os.path.abspath(filename)), "error")
            elif filename.endswith('.csv'):
                if analyse_csv(os.path.abspath(filename)) == 'Low':
                    flash('Student cheating risk: ' + analyse_csv(os.path.abspath(filename)))
                else:
                    flash('Student cheating risk: ' + analyse_csv(os.path.abspath(filename)), "error")
    return render_template("gaze_analyse.html", user=current_user)
