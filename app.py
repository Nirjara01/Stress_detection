from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import joblib
import numpy as np
import pandas as pd
from validate_gb_voting import StressValidationSystem  # Import validation system

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load ML model and scaler
model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')
validation_model = StressValidationSystem()  # Initialize validation system

# Database configuration

def get_db():
    conn = sqlite3.connect('stress_detection5.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                attempts_left INTEGER DEFAULT 20
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                stress_level TEXT NOT NULL,
                recommendations TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        db.commit()

init_db()

# Quiz questions and recommendations
quiz_questions = [
    "How often do you feel stressed at work?",
    "Do you experience trouble sleeping due to anxiety?",
    "How frequently do you feel overwhelmed by responsibilities?",
    "How often do you experience difficulty concentrating due to stress?",
    "Do you find it hard to relax even during leisure time?"
]

recommendations = {
    'Low': {
        'general': "Great job managing stress! Maintain your healthy habits.",
        'specific': [
            "Continue current work-life balance",
            "Maintain sleep routine",
            "Keep prioritizing tasks",
            "Stay focused with current strategies",
            "Continue relaxation practices"
        ]
    },
    'Moderate': {
        'general': "Consider implementing stress reduction techniques:",
        'specific': [
            "Try time management techniques",
            "Practice relaxation before bed",
            "Break tasks into smaller steps",
            "Take regular short breaks",
            "Practice deep breathing exercises"
        ]
    },
    'High': {
        'general': "Recommend professional consultation and:",
        'specific': [
            "Seek workload management advice",
            "Consult sleep specialist",
            "Practice delegation techniques",
            "Try mindfulness meditation",
            "Join stress management program"
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        try:
            db = get_db()
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                      (username, password))
            db.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!')
        finally:
            db.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['attempts_left'] = user['attempts_left']
            return redirect(url_for('quiz'))
        flash('Invalid credentials!')
    return render_template('login.html')


@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    db = get_db()
    user = db.execute('SELECT username FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    db.close()

    if session.get('attempts_left', 0) <= 0:
        flash('No attempts remaining!')
        return redirect(url_for('index'))

    if request.method == 'POST':
        responses = [request.form[f'q{i}'] for i in range(1, 6)]
        response_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
        numeric_responses = [response_mapping[resp] for resp in responses]

        scaled_data = scaler.transform([numeric_responses])
        prediction = model.predict(scaled_data)[0]
        stress_level = ['Low', 'Moderate', 'High'][prediction]

        validation_result = validation_model.validate_with_gb_voting(responses)

        db = get_db()
        db.execute('INSERT INTO results (user_id, stress_level, recommendations) VALUES (?, ?, ?)',
                  (session['user_id'], stress_level, str(recommendations[stress_level])))
        db.execute('UPDATE users SET attempts_left = attempts_left - 1 WHERE id = ?',
                  (session['user_id'],))
        db.commit()
        db.close()

        session['attempts_left'] -= 1

        session['stress_level'] = stress_level
        session['validation_result'] = validation_result

        return redirect(url_for('result'))

    return render_template('quiz.html', questions=quiz_questions, username=user['username'])


@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    # Fetch aggregated stress levels
    pie_cursor = db.execute('SELECT stress_level, COUNT(*) as count FROM results GROUP BY stress_level')
    pie_data = {row['stress_level']: row['count'] for row in pie_cursor.fetchall()}

    # Fetch detailed results with usernames
    results_cursor = db.execute('''
        SELECT users.username, results.stress_level, results.recommendations
        FROM results
        JOIN users ON users.id = results.user_id
        ORDER BY results.id DESC
    ''')
    results = results_cursor.fetchall()
    db.close()

    return render_template('admin_dashboard.html', pie_data=pie_data, results=results)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    user = db.execute('SELECT username FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    result = db.execute('SELECT * FROM results WHERE user_id = ? ORDER BY id DESC LIMIT 1', (session['user_id'],)).fetchone()
    db.close()

    if result:
        level = result['stress_level']
        recs = recommendations[level]
    else:
        level = "Not Available"
        recs = {'general': "Take the quiz to get results.", 'specific': []}

    return render_template('dashboard.html',
                           username=user['username'],
                           stress_level=level,
                           recommendations=recs)



@app.route('/result', methods=['GET', 'POST'])
def result():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    db = get_db()
    user = db.execute('SELECT username FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    db.close()


    stress_level = session.get('stress_level', 'Moderate')
    validation_result = session.get('validation_result', 'Moderate')

    if request.method == 'POST':
        return redirect(url_for('validated_result'))

    return render_template('result.html',
                         stress_level=stress_level,
                         recommendations=recommendations[stress_level], username=user['username'])

@app.route('/validated_result')
def validated_result():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    stress_level = session.get('stress_level', 'Moderate')
    validation_result = session.get('validation_result', 'Moderate')

    return render_template('validated_result.html',
                           stress_level=stress_level,
                           validation_result=validation_result)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
