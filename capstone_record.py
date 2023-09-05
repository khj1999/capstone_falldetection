from flask import Flask
from flask import render_template, send_file, Response

app = Flask(__name__)

@app.route("/")
def video():
    return render_template("video record.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)