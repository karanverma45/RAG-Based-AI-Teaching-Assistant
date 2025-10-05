from flask import Flask, render_template, request, jsonify, session
from process_incoming import answer_query
import os
from datetime import datetime
import mysql.connector
from mysql.connector import errors as mysql_errors

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')


DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '127.0.0.1'),
    'port': int(os.environ.get('DB_PORT', '3306')),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': os.environ.get('DB_NAME', 'rag_app'),
}


def get_db_connection():
    conn = mysql.connector.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255),
            password_hash VARCHAR(255) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            question TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            CONSTRAINT fk_questions_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )
    conn.commit()
    conn.close()


@app.before_request
def setup():
    try:
        init_db()
    except Exception:
        # Avoid blocking requests if migration fails; surface error on write ops
        pass


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# API: Ask a question
@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({"error": "Query is required."}), 400
    try:
        response_text = answer_query(query)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Save question if user logged in
    user_id = session.get('user_id')
    if user_id:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO questions (user_id, question, created_at) VALUES (%s, %s, %s)",
            (user_id, query, datetime.utcnow())
        )
        conn.commit()
        conn.close()

    return jsonify({"answer": response_text})


# API: Signup
@app.route('/signup', methods=['POST'])
def signup():
    from werkzeug.security import generate_password_hash

    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = (data.get('password') or '').strip()
    name = (data.get('name') or '').strip()
    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    password_hash = generate_password_hash(password)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, name, password_hash) VALUES (%s, %s, %s)",
            (email, name, password_hash)
        )
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
    except mysql_errors.IntegrityError:
        return jsonify({"error": "Email already registered."}), 409

    session['user_id'] = user_id
    session['user_email'] = email
    session['user_name'] = name
    return jsonify({"ok": True})


# API: Login
@app.route('/login', methods=['POST'])
def login():
    from werkzeug.security import check_password_hash

    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = (data.get('password') or '').strip()
    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email, name, password_hash FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    conn.close()

    if not row or not check_password_hash(row[3], password):
        return jsonify({"error": "Invalid credentials."}), 401

    session['user_id'] = row[0]
    session['user_email'] = row[1]
    session['user_name'] = row[2]
    return jsonify({"ok": True})


# API: Logout
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"ok": True})


# API: Fetch user history
@app.route('/api/history', methods=['GET'])
def api_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"questions": []})
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT question, created_at FROM questions WHERE user_id = %s ORDER BY id DESC LIMIT 50",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return jsonify({
        "questions": [
            {"question": r[0], "created_at": r[1].isoformat() if r[1] else None} for r in rows
        ]
    })


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)


