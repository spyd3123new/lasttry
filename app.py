from flask import Flask, request, jsonify , render_template
from runprediction import predictionsz


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    data = request.json
    content = data.get('content')
    recommended_asans = predictionsz(content)
    print("Received content:", content)
    return jsonify({"recommended_asans": recommended_asans})

if __name__ == '__main__':
    app.run(debug=True)
