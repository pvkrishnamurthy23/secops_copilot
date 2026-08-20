from flask import Flask, request, jsonify
import sqlite3
import subprocess
import pickle
import hashlib
import os

app = Flask(__name__)

# VULNERABILITY: Hardcoded credentials / secrets
DB_PASSWORD = "SuperSecret123!"
AWS_ACCESS_KEY = "AKIAFAKE1234567890"
AWS_SECRET_KEY = "fakeSecretKeyForTrainingOnly123456789"
API_TOKEN = "sk-test-THIS_IS_A_FAKE_TOKEN"
ADMIN_PASSWORD = "admin123"

DATABASE = "users.db"


def get_db():
    return sqlite3.connect(DATABASE)


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    # VULNERABILITY: SQL Injection
    query = (
        "SELECT * FROM users WHERE username = '"
        + username
        + "' AND password = '"
        + password
        + "'"
    )

    conn = get_db()
    result = conn.execute(query).fetchone()
    conn.close()

    if result:
        return jsonify({"status": "success", "user": username})

    return jsonify({"status": "failed"}), 401


@app.route("/ping")
def ping():
    host = request.args.get("host", "localhost")

    # VULNERABILITY: OS Command Injection
    output = subprocess.check_output(
        "ping -c 1 " + host,
        shell=True,
        text=True
    )

    return output


@app.route("/file")
def read_file():
    filename = request.args.get("name")

    # VULNERABILITY: Path Traversal
    path = "/tmp/uploads/" + filename

    with open(path, "r") as f:
        return f.read()


@app.route("/deserialize", methods=["POST"])
def deserialize():
    data = request.get_data()

    # VULNERABILITY: Unsafe Deserialization
    obj = pickle.loads(data)

    return str(obj)


@app.route("/hash-password", methods=["POST"])
def hash_password():
    password = request.form.get("password")

    # VULNERABILITY: Weak password hashing
    password_hash = hashlib.md5(password.encode()).hexdigest()

    return jsonify({"hash": password_hash})


@app.route("/admin")
def admin():
    password = request.args.get("password")

    # VULNERABILITY: Hardcoded password
    # VULNERABILITY: Password passed in URL/query string
    if password == ADMIN_PASSWORD:
        return "Welcome Admin"

    return "Access denied", 403


@app.route("/debug")
def debug_info():
    # VULNERABILITY: Sensitive information disclosure
    return jsonify({
        "database_password": DB_PASSWORD,
        "aws_access_key": AWS_ACCESS_KEY,
        "aws_secret_key": AWS_SECRET_KEY,
        "environment": dict(os.environ)
    })


@app.route("/download")
def download():
    url = request.args.get("url")

    # VULNERABILITY:
    # User-controlled URL could result in SSRF if used with requests.get()
    import requests
    response = requests.get(url)

    return response.text


if __name__ == "__main__":
    # VULNERABILITY: Debug mode enabled
    # VULNERABILITY: Listening on all interfaces
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
