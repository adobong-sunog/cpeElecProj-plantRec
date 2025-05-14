import os
import uuid
import sqlite3
import numpy as np
from functools import wraps
from flask import Flask, g, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_ROOT = os.path.join(STATIC_DIR, 'uploads')
os.makedirs(UPLOAD_ROOT, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=os.path.join(BASE_DIR, 'templates'))
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret')

model = load_model(os.path.join(BASE_DIR, 'plant_disease_model.h5'))

# config
DATABASE = os.path.join(BASE_DIR, 'users.db')
MAX_HISTORY = 5
ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS = os.environ.get('ADMIN_PASS', 'admin')
CLASS_LABELS = ['Healthy', 'Powdery', 'Rust']
ALLOWED_EXTS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db:
        db.close()

def init_db():
    db = get_db()
    db.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0
                  )""")
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    img_path TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                  )""")
    db.commit()
    cur = db.execute("SELECT COUNT(*) AS cnt FROM users WHERE is_admin=1").fetchone()
    if cur['cnt'] == 0:
        db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
                   (ADMIN_USER, generate_password_hash(ADMIN_PASS)))
        db.commit()

with app.app_context():
    init_db()

def login_required(f):
    @wraps(f)
    def w(*args, **kw):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kw)
    return w

def preprocess(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)

@app.route('/')
@login_required
def index():
    db = get_db()
    rows = db.execute(
        "SELECT img_path, prediction, confidence, timestamp FROM predictions "
        "WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (session['user_id'], MAX_HISTORY)
    ).fetchall()
    default_icon_path = 'rscrs/icon.png'
    return render_template('index.html', history=rows, icon=default_icon_path)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # log prediction process
    app.logger.info("Starting prediction process")
    app.logger.info(f"User ID: {session['user_id']}")
    
    f = request.files.get('file')
    if not f or f.filename == '':
        app.logger.error("No file provided")
        flash('No file', 'error')
        return redirect(url_for('index'))
    
    app.logger.info(f"Received file: {f.filename}")
    
    if not allowed_file(f.filename):
        app.logger.error(f"Invalid file type: {f.filename}")
        flash('Bad file type', 'error')
        return redirect(url_for('index'))

    # set unique filename
    orig = secure_filename(f.filename)
    fn = f"{uuid.uuid4().hex}_{orig}"
    app.logger.info(f"Original filename: {orig}, Generated filename: {fn}")

    # Per-user folder
    user_dir = os.path.join(UPLOAD_ROOT, str(session['user_id']))
    app.logger.info(f"User directory path: {user_dir}")
    
    # check if upload directory exists
    if not os.path.exists(UPLOAD_ROOT):
        app.logger.error(f"Uploads root directory does not exist: {UPLOAD_ROOT}")
        try:
            os.makedirs(UPLOAD_ROOT, exist_ok=True)
            app.logger.info(f"Created uploads root directory: {UPLOAD_ROOT}")
        except Exception as e:
            app.logger.error(f"Failed to create uploads directory: {e}")
            flash('Server error', 'error')
            return redirect(url_for('index'))

    # create user directory if it doesn't exist
    try:
        os.makedirs(user_dir, exist_ok=True)
        app.logger.info(f"Ensured user directory exists: {user_dir}")
    except Exception as e:
        app.logger.error(f"Failed to create user directory: {e}")
        flash('Server error', 'error')
        return redirect(url_for('index'))

    # verify if directory exists
    if not os.path.exists(user_dir):
        app.logger.error(f"User directory still doesn't exist after creation attempt: {user_dir}")
        flash('Server error', 'error')
        return redirect(url_for('index'))

    abs_path = os.path.join(user_dir, fn)
    app.logger.info(f"Will save file to: {abs_path}")
    
    # check for write permission
    try:
        # create test file
        test_path = os.path.join(user_dir, "test_write_permission.txt")
        with open(test_path, 'w') as test_file:
            test_file.write("testing write permission")
        os.remove(test_path)
        app.logger.info(f"Write permission to user directory confirmed")
    except Exception as e:
        app.logger.error(f"No write permission to user directory: {e}")
        flash('Server error - no write permission', 'error')
        return redirect(url_for('index'))

    try:
        f.save(abs_path)
        app.logger.info(f"File saved to {abs_path}")
        
        # chekc if file exists
        if os.path.exists(abs_path):
            app.logger.info(f"File exists at {abs_path}, size: {os.path.getsize(abs_path)} bytes")
        else:
            app.logger.error(f"File does not exist at {abs_path} after save operation")
            flash('File save failed', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"Error saving file: {str(e)}")
        flash('Error saving file', 'error')
        return redirect(url_for('index'))
    
    try:
        preds = model.predict(preprocess(abs_path))[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx] * 100)
        
        # Add confidence threshold check
        confidence_threshold = 70.0  # Adjust this value based on testing
        if conf < confidence_threshold:
            label = "Not a leaf or unrecognized"
        else:
            label = CLASS_LABELS[idx]
            
        app.logger.info(f"Prediction: {label}, Confidence: {conf}%")
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        flash('Error during prediction', 'error')
        return redirect(url_for('index'))

    rel = f"uploads/{session['user_id']}/{fn}"
    app.logger.info(f"Relative path for database: {rel}")
    
    db = get_db()
    try:
        db.execute("INSERT INTO predictions (user_id, img_path, prediction, confidence) VALUES (?, ?, ?, ?)",
                (session['user_id'], rel, label, conf))
        db.commit()
        app.logger.info("Prediction saved to database")
    except Exception as e:
        app.logger.error(f"Error saving prediction to database: {str(e)}")
    

    rows = db.execute(
        "SELECT img_path, prediction, confidence, timestamp FROM predictions "
        "WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (session['user_id'], MAX_HISTORY)
    ).fetchall()

    return render_template('result.html',
                        prediction=label,
                        confidence=f"{conf:.2f}",
                        img_path=rel,
                        history=rows)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        u = request.form['username'].strip()
        p = request.form['password']
        db = get_db()

        if form_type == 'login':    
            row = db.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
            if row and check_password_hash(row['password'], p):
                session.clear()
                session['user_id'] = row['id']
                session['username'] = row['username']
                session['is_admin'] = bool(row['is_admin'])
                return redirect(url_for('index'))
            flash('Invalid', 'error')
            
        elif form_type == 'register':
            if not u or not p:
                flash('User/pass required', 'error')
            else:
                db = get_db()
            try:
                db.execute("INSERT INTO users(username, password) VALUES(?, ?)",
                           (u, generate_password_hash(p)))
                db.commit()
                flash('Registered', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Taken', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin():
    if not session.get('is_admin'):
        flash('Admin only', 'error')
        return redirect(url_for('index'))
    users = get_db().execute("SELECT id, username, is_admin FROM users").fetchall()
    return render_template('admin.html', users=users)

@app.route('/admin/add_user', methods=['POST'])
@login_required
def add_user():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    u = request.form['username'].strip()
    p = request.form['password']
    ia = 1 if request.form.get('is_admin') == 'on' else 0
    if not u or not p:
        flash('User/pass required', 'error')
    else:
        db = get_db()
        try:
            db.execute("INSERT INTO users(username, password, is_admin) VALUES(?, ?, ?)",
                       (u, generate_password_hash(p), ia))
            db.commit()
            flash('Added', 'success')
        except sqlite3.IntegrityError:
            flash('Taken', 'error')
    return redirect(url_for('admin'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    if user_id == session['user_id']:
        flash("Can't delete self", 'error')
    else:
        get_db().execute("DELETE FROM users WHERE id=?", (user_id,))
        get_db().commit()
        flash('Deleted', 'info')
    return redirect(url_for('admin'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)