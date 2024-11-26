import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from werkzeug.utils import secure_filename
from model_functions import predict_gesture

app = Flask(__name__)
model = load_model('./hagrid1.keras')

class_names = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace_inverted', 'peace',
    'rock', 'stop_inverted', 'stop', 'three', 'three2', 'two_up', 'two_up_inverted', 'no_gesture'
]

UPLOAD_FOLDER = 'static/uploaded_images/'
WEBCAM_FOLDER = 'static/webcam_images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['WEBCAM_FOLDER'] = WEBCAM_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')  # Nova rota adicionada

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Processa a imagem
        frame = cv2.imread(file_path)
        gesture, frame_with_box = predict_gesture(frame, model, class_names)

        # Adiciona a previsão do gesto na imagem
        cv2.putText(frame_with_box, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Salva a imagem processada
        result_filename = 'resultado_' + filename
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_image_path, frame_with_box)

        # Gera o caminho público para exibição no navegador
        image_path = url_for('static', filename=f'uploaded_images/{result_filename}')

        # Renderiza o main.html com os resultados
        return render_template('main.html', gesture=gesture, image_path=image_path)


@app.route('/webcam_capture', methods=['POST'])
def webcam_capture():
    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({'error': 'No image data received'}), 400

    # Remove o prefixo 'data:image/jpeg;base64,' e decodifica o base64
    image_data = image_data.split(',')[1]

    # Converte a imagem de base64 para formato OpenCV
    frame = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Failed to process image'}), 400

    # Predição do gesto
    gesture, frame_with_box = predict_gesture(frame, model, class_names)

    # Adiciona a predição na imagem
    cv2.putText(frame_with_box, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Codifica a imagem processada em base64
    _, buffer = cv2.imencode('.jpg', frame_with_box)  # Aqui codificamos a imagem para JPEG
    if buffer is None:
        return jsonify({'error': 'Failed to encode image'}), 400

    frame_base64 = base64.b64encode(buffer).decode('utf-8')  # Codifica para base64

    # Retorna a imagem e o gesto em formato JSON
    return jsonify({
        'gesture': gesture,
        'image_path': f'data:image/jpeg;base64,{frame_base64}'  # Retorna a imagem em base64
    })


if __name__ == '__main__':
    app.run(debug=True)
