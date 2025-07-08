import os
import json
import base64
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from attention_explainer import generate_final_explanation

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
MASKS_DIR = os.path.join(BASE_DIR, 'maschere_lesioni_colorate_504x504') 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = ["No apparent retinopathy", "Mild non-proliferative retinopathy", "Moderate NPDR", "Severe NPDR", "Proliferative diabetic retinopathy"]
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_image_path)

            predicted_class_index, confidence, final_image_path, attention_data_path, lesion_data_path = generate_final_explanation(
                uploaded_image_path, MASKS_DIR, app.config['RESULTS_FOLDER']
            )

            if predicted_class_index is not None:
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                return render_template('results.html',
                                       predicted_class=f"Predicted Class: {predicted_class_name}",
                                       confidence=f"Confidence: {confidence:.2%}",
                                       original_filename=filename,
                                       analysis_filename=os.path.basename(final_image_path),
                                       attention_data_filename=os.path.basename(attention_data_path),
                                       lesion_data_filename=os.path.basename(lesion_data_path)
                )
            else: return "Error processing image.", 500
        else: return "Invalid file type.", 400
    return render_template('upload.html')

@app.route('/annotate/<original_filename>/<attention_data_filename>/<lesion_data_filename>')
def annotate(original_filename, attention_data_filename, lesion_data_filename):
    attention_data, lesion_data, image_data_url = [], [], None
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        image_data_url = f"data:image/png;base64,{encoded_string}"
        with open(os.path.join(app.config['RESULTS_FOLDER'], attention_data_filename), 'r') as f:
            attention_data = json.load(f)
        with open(os.path.join(app.config['RESULTS_FOLDER'], lesion_data_filename), 'r') as f:
            lesion_data = json.load(f)
    except Exception as e:
        print(f"Errore nel preparare i dati per la pagina di annotazione: {e}")

    return render_template('annotate.html',
                           original_image_data_url=image_data_url,
                           attention_data=attention_data,
                           lesion_data=lesion_data
                           )

@app.route('/uploads_display/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)