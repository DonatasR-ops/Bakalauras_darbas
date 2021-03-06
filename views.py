from flask import Blueprint, render_template, Response, send_file, redirect, url_for, request, Flask, flash
from flask_login import login_required, current_user
import glob
import os
from gaze_tracker import GazeTracker
from werkzeug.utils import secure_filename
GazeTracker = GazeTracker()
class ApplicationWindows:

    views = Blueprint('views', __name__)

    ALLOWED_EXTENSIONS = {'csv', 'mp4'}

    @views.route('/')
    def home(self):
        return render_template("home.html", user=current_user)


    @views.route('/gaze-real', methods=['GET', 'POST'])
    def gaze_real(self):
        return render_template("gaze_real.html", user=current_user)


    @views.route('/download')
    def download_file(self):
        list_of_files = glob.glob('*.csv')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        p = latest_file
        return send_file(p, as_attachment=True)


    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ApplicationWindows.ALLOWED_EXTENSIONS


    def gen_frames_algorithm(self):  # generate frame by frame from camera
        time_right = 0
        time_center = 0
        time_left = 0
        while True:
            success, frame = GazeTracker.camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = GazeTracker.detector(gray)
            for face in faces:
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)
                landmarks = GazeTracker.predictor(gray, face)
                gaze_ratio_left_eye = GazeTracker.get_gaze_ratio(frame, gray, [36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = GazeTracker.get_gaze_ratio(frame, gray, [42, 43, 44, 45, 46, 47], landmarks)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
                new_frame = np.zeros((500, 500, 3), np.uint8)
                start = time.time()
                print(gaze_ratio)
                if GazeTracker.gaze_ratio >= GazeTracker.gaze_right:
                    cv2.putText(frame, "LEFT", (x - 100, y), GazeTracker.font, 1, (0, 0, 255), 2)
                    new_frame[:] = (0, 0, 255)
                    end1 = time.time()
                    time_left += (end1 - start) * 100
                elif GazeTracker.gaze_right > gaze_ratio > GazeTracker.gaze_left:
                    cv2.putText(frame, "CENTER", (x - 100, y), font, 1, (0, 0, 255), 2)
                    end2 = time.time()
                    time_center += (end2 - start) * 1000
                else:
                    new_frame[:] = (255, 0, 0)
                    cv2.putText(frame, "RIGHT", (x - 100, y), font, 1, (0, 0, 255), 2)
                    end3 = time.time()
                    time_right += (end3 - start) * 100
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


    def gen_frames(self):
        while True:
            success, frame = GazeTracker.camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    @views.route('/video_feed_base')
    def video_feed_base(self):
        return Response(ApplicationWindows.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @views.route('/video_feed')
    def video_feed(self):
        return Response(ApplicationWindows.gen_frames_algorithm(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @views.route("/gaze-analyse", methods=["GET", "POST"])
    def gaze_analyse(self):
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if not ApplicationWindows.allowed_file(file.filename):
                flash('Wrong file format', "error")
                return redirect(request.url)
            if file and ApplicationWindows.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if filename.endswith('.mp4'):
                    result = GazeTracker.analyse_video(os.path.abspath(filename))
                    print(result)
                    if result == 'Low':
                        flash('Student cheating risk: ' + result)
                    else:
                        flash('Student cheating risk: ' + result, "error")
                elif filename.endswith('.csv'):
                    if ApplicationWindows.analyse_csv(os.path.abspath(filename)) == 'Low':
                        flash('Student cheating risk: ' + ApplicationWindows.analyse_csv(os.path.abspath(filename)))
                    else:
                        flash('Student cheating risk: ' + ApplicationWindows.analyse_csv(os.path.abspath(filename)), "error")
        return render_template("gaze_analyse.html", user=current_user)
