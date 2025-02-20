from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
import pyotp
import qrcode
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Connect to SQLite database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Setup the database (Run this once to create the table)
def setup_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            password TEXT NOT NULL,
            secret TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Home page with Register and Login options
@app.route('/')
def home():
    return render_template('home.html')

# Registration page
@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return 'Error: Missing User ID or Password', 400

        # Generate a unique secret for Google Authenticator
        secret = pyotp.random_base32()

        # Save the user in the database with the secret
        conn = get_db_connection()
        conn.execute('INSERT INTO users (user_id, password, secret) VALUES (?, ?, ?)', (username, password, secret))
        conn.commit()
        conn.close()

        # Create a QR code for Google Authenticator
        totp = pyotp.TOTP(secret)
        otp_url = totp.provisioning_uri(username, issuer_name="E-Authorization")
        img = qrcode.make(otp_url)
        buf = BytesIO()
        img.save(buf)
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return render_template('qr_code.html', img_data=img_base64)

    return render_template('register.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE user_id = ? AND password = ?', (username, password)).fetchone()
        conn.close()

        if user:
            # Store the secret in the session for OTP verification
            session['user_id'] = username
            session['secret'] = user['secret']
            return redirect('/otp')
        else:
            return 'Invalid credentials, please try again.'

    return render_template('login.html')

# OTP verification page
@app.route('/otp', methods=['GET', 'POST'])
def otp_page():
    if request.method == 'POST':
        otp = request.form.get('otp')
        secret = session.get('secret')

        # Verify the OTP
        totp = pyotp.TOTP(secret)
        if totp.verify(otp):
            return render_template('success.html', message="Congratulations, you have logged in successfully!")
        else:
            return 'Invalid OTP, please try again.'

    return render_template('otp.html')

# Run the app on port 5004
if __name__ == '__main__':
    setup_db()  # Ensure the database is set up before starting the app
    app.run(debug=True, port=5004)