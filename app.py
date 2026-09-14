import os
from flask import Flask, jsonify, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify(
        status="success", message="Application is running smoothly on Vercel!"
    )


@app.route("/health")
def health():
    return jsonify(status="healthy"), 200


# Error handler for 404 routes within Flask
@app.errorhandler(404)
def not_found(error):
    return jsonify(error="Endpoint not found", status_code=404), 404


if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
