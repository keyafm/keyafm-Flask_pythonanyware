from flask import Flask, request, jsonify
import pickle
import numpy as np
import git

app = Flask(__name__)

# Cargamos el modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/git_update', methods=['POST'])
def git_update():
    repo = git.Repo('./pythonanywhere')
    origin = repo.remotes.origin
    repo.create_head('main',
                     origin.refs.main).set_tracking_branch(origin.refs.main).checkout()
    origin.pull()
    return '', 200
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(np.array(data['example']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run()

