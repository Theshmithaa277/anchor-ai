"""
auth.py — Anchor AI V2 authentication.

A self-contained Flask Blueprint providing the three-stage onboarding
journey (welcome -> login/signup -> chatbot) plus logout. Owns the
Bcrypt and LoginManager instances so app.py just has to init_app() them
and register this blueprint — no auth logic lives in app.py itself.

Routes (mounted with url_prefix="/auth" in app.py):
    GET  /auth/welcome   - full-screen editorial welcome screen
    GET  /auth/login     - login form
    POST /auth/login     - authenticate
    GET  /auth/signup    - signup form
    POST /auth/signup    - create account (and log the user straight in)
    GET  /auth/logout    - clear session, back to welcome
"""

import re

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)

from models import User, db

bcrypt = Bcrypt()

login_manager = LoginManager()
login_manager.login_view = "auth.welcome"

# Endpoints that are called via fetch() from anchor_ai.html and must always
# get a JSON response (never an HTML redirect to the login page), so the
# existing frontend's parseResponse()/error handling keeps working exactly
# as it did before auth was added.
_API_PATHS = {"/ingest", "/ask"}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8

auth_bp = Blueprint("auth", __name__, template_folder="templates")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    if request.path in _API_PATHS:
        return jsonify({
            "status": "error",
            "message": "Please log in to continue.",
        }), 401
    return redirect(url_for(login_manager.login_view, next=request.path))


# ---------------------------------------------------------------------
# WELCOME
# ---------------------------------------------------------------------

@auth_bp.route("/welcome")
def welcome():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    return render_template("welcome.html")


# ---------------------------------------------------------------------
# LOGIN
# ---------------------------------------------------------------------

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template("login.html")

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    error = None
    if not email or not password:
        error = "Please enter both email and password."
    else:
        user = User.query.filter_by(email=email).first()
        if not user or not bcrypt.check_password_hash(user.password_hash, password):
            error = "Incorrect email or password."

    if error:
        flash(error, "error")
        return render_template("login.html", email=email), 401

    login_user(user, remember=True)
    next_path = request.args.get("next")
    return redirect(next_path or url_for("home"))


# ---------------------------------------------------------------------
# SIGNUP
# ---------------------------------------------------------------------

@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template("signup.html")

    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    error = None
    if not name:
        error = "Please enter your name."
    elif not EMAIL_RE.match(email):
        error = "Please enter a valid email address."
    elif len(password) < MIN_PASSWORD_LENGTH:
        error = f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    elif User.query.filter_by(email=email).first() is not None:
        error = "An account with that email already exists."

    if error:
        flash(error, "error")
        return render_template("signup.html", name=name, email=email), 400

    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(name=name, email=email, password_hash=password_hash)
    db.session.add(user)
    db.session.commit()

    login_user(user, remember=True)
    return redirect(url_for("home"))


# ---------------------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------------------

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.welcome"))
