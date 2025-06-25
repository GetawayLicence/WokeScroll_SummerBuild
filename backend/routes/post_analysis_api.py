
from flask import Blueprint, Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import os
from openai import OpenAI

post_analysis_bp = Blueprint("post_analysis", __name__)

# --- Initialization ---    

load_dotenv()
# <FIGURE OUT WHY .ENV ISNT WORKING THIS IS KINDA DANGEROUS>
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["OPENAI_API_KEY"],
)
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

# --- Helper Functions ---

def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]
    return {"label": result["label"], "score": result["score"]}

from transformers import pipeline

summarizer = pipeline("summarization")

def summarize_text(text):
    cleaned = text.strip()
    word_count = len(cleaned.split())

    if word_count < 10:
        return cleaned  

    try:
        result = summarizer(
            cleaned[:1024],
            max_length=100,
            min_length=30,
            do_sample=False
        )
        return result[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {str(e)}"


def tone_analysis(text):
    response = client.chat.completions.create(
        model="openai/gpt-4o",  # or "gpt-4o" if your account supports it
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in tone analysis."
                    "Given a piece of text, return only a JSON array of lowercase tone adjectives. "
                    "Examples: [\"formal\", \"sarcastic\"]. "
                    "Do not explain anything. Just return the list."
                )
            },
            {
                "role": "user",
                "content": f"""
Analyze the following text for the tone — classify it with adjectives such as: formal, casual, persuasive, aggressive, apologetic, cheerful, sarcastic, etc.

Text:
\"\"\"{text}\"\"\"
"""
            }
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1
    )
    # Try to parse JSON safely
    try:
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Parsing error:", e)
        return []


def manipulative_language_analysis(text):
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "system",
                "content": ( 
                "You are a language analysis expert." 
                "Identify and explain any manipulative or emotionally exploitative phrases."
                "Given a piece of text, return only a bulleted list of phrases containing (negative) manipulative language and a brief descriptions for each phrase."
                
                )
            },
            {
                "role": "user",
                "content": f"""
Analyze whether the text contains any manipulative or emotionally exploitative language (e.g., fear-mongering, guilt-tripping, exaggeration, peer pressure). 
If yes, explain briefly for each phrase. 

Text:
\"\"\"{text}\"\"\"
"""
            }
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1
    )
    return response.choices[0].message.content.strip()

def calculate_vibescore(sentiment, manipulation_text):
    score = 90
    if sentiment["label"] == "POSITIVE":
        score += sentiment["score"] * 30
    elif sentiment["label"] == "NEGATIVE":
        score -= sentiment["score"] * 30

    if any(word in manipulation_text.lower() for word in ["manipulative", "exploitative", "fear", "guilt"]):
        score -= 10

    return max(0, min(100, round(score, 2)))

# --- API Route ---

@post_analysis_bp.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        sentiment = analyze_sentiment(text)
        summary = summarize_text(text)
        tone = tone_analysis(text)
        manipulation = manipulative_language_analysis(text)
        vibe_score = calculate_vibescore(sentiment, manipulation)

        return jsonify({
            "summary": summary,
            #"sentiment": sentiment,
            "tone": tone,
            "manipulation_analysis": manipulation,
            "vibe_score": vibe_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@post_analysis_bp.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})





