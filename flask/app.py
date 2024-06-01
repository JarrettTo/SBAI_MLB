import sys
import os
from datetime import date
import json
from flask import Flask, render_template, request
from functools import lru_cache

root_dir = os.getenv('ROOT_DIR', 'default_fallback_path')

sys.path.append(root_dir)
from predictions import main

app = Flask(__name__)

@app.route("/")
def index():
    return "hello"

@app.route("/predict") 
def predict():
    res=main()
    return res

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=4000)
