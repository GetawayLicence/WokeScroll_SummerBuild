from flask import Flask
from routes.fake_news_api import fake_news_bp
from routes.post_analysis_api import post_analysis_bp

app = Flask(__name__)

@app.route("/")
def home():
    return "Fake News Detector API is running."

app.register_blueprint(fake_news_bp)
app.register_blueprint(post_analysis_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

