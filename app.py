from flask import Flask, render_template, request, jsonify
from chatbot.bot import process_query

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("query", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    response = process_query(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
