import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from pyngrok import ngrok
import pandas as pd
from process_files import process_files

# Flask app setup
app = Flask(__name__)

# Set up ngrok with your auth token
NGROK_AUTH_TOKEN = "SECRET_TOKEN"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)


@app.route("/")
def home():
    return render_template('app.html')

@app.route("/upload", methods=["POST"])
def upload_files():
    print("in upload_files")
    try:
        print("enter try")
        #if 'dat_file' not in request.files or 'csv_file' not in request.files:
            #return jsonify({"error": "Missing EEG or facial data file."}), 400
        print("request files")
        dat_file = request.files['dat_file']
        csv_file = request.files['csv_file']
        print("check filenames")
        #if dat_file.filename == '' or csv_file.filename == '':
            #return jsonify({"error": "No file selected."}), 400
        print("get files and send to process")
        output = process_files(dat_file, csv_file)
        print("get output in upload_files")
        print(output["predictions"],output["groups"],output["emotions"])
        response = {
            "modelResults": output["predictions"],
            "groups": output["groups"],
            "emotions": output["emotions"]
        }
        print("response sent to front end")
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Error during file processing: {str(e)}"}), 500

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f"ngrok tunnel available at: {public_url}")

    app.run(port=5000)