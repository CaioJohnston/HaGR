import base64
import os
import cv2
import threading
import time
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, redirect, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("./YOLOv10x_gestures.pt")

UPLOAD_FOLDER = 'static/uploaded_images/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo inválido'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        frame = cv2.imread(file_path)
        results = model(frame)
        frame_with_boxes = results[0].plot()
        
        result_filename = 'resultado_' + filename
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_image_path, frame_with_boxes)
        
        image_url = url_for('static', filename=f'uploaded_images/{result_filename}', _external=True)
        return jsonify({'image_path': image_url})

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        frame_with_boxes = results[0].plot()

        _, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
