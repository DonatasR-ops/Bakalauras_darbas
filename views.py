from flask import Blueprint, render_template
from flask_login import login_required, current_user

views = Blueprint('views', __name__)


@views.route('/')
def home():
    return render_template("home.html", user=current_user)


@views.route('/gaze-real')
def gaze_real():
    return render_template("gaze_real.html", user=current_user)


@views.route('/gaze-analyse')
def gaze_analyse():
    return render_template("gaze_analyse.html", user=current_user)