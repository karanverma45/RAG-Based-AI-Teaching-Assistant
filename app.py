from flask import Flask, render_template, request, jsonify
from process_incoming import answer_query

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


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
    return jsonify({"answer": response_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


