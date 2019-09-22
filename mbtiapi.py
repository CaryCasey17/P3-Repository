from flask import Flask, jsonify, abort, request, make_response, url_for, g
import uuid
import random
import base64
from werkzeug.contrib.fixers import ProxyFix
import pickle
from sklearn.linear_model import LogisticRegression
import myersbriggs as mb

# initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
app.config['HOST'] = '127.0.0.1'
app.config['DATABASE'] = 'web'
app.config['USER_NAME'] = 'root'
app.config['PASSWORD'] = 'root'


@app.route('/')
def api_root():
  return "Welcome"

@app.route('/mbti', methods=['GET'])

def classify():
    comment = request.args['comment']
    X = comment
    y = mb.mbti_predict(X)

    obj = {"MBTI": y}
    
    return jsonify(obj)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=10015, debug= True)
