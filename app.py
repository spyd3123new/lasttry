from flask import Flask, request, jsonify
from runprediction import predictionsz
from flask_cors import CORS ,cross_origin

app = Flask(__name__, static_folder='my-app/build', static_url_path='')
CORS(app)

@app.route('/api', methods=['POST'])
@cross_origin()
def api():
    data = request.json
    content = data.get('content')
    recommended_asans = predictionsz(content)
    print("Received content:", content)
    return jsonify({"recommended_asans": recommended_asans})

@app.route('/')
@cross_origin()
def serve():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
