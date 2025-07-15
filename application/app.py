# =============================================================================
#                        FLASK WEB APPLICATION
#
# This script is the main entry point for a Flask web application that provides
# an interface for a diabetic retinopathy classifier. It allows users to upload
# a retinal image, receive a classification, and view a visual explanation
# of the model's decision using attention maps.
#
# Main Features:
# - File upload with validation.
# - Backend processing using the `generate_final_explanation` function.
# - A results page displaying the prediction and analysis image.
# - An interactive annotation/explanation view using client-side JavaScript.
# =============================================================================

import os
import json
import base64
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Import the main processing function from the explainer module.
from attention_explainer import generate_final_explanation

# --- Application Setup and Configuration ---
app = Flask(__name__)

# Define base directories and folder paths.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
# Path to ground truth lesion masks for comparison.
MASKS_DIR = os.path.join(BASE_DIR, 'maschere_lesioni_colorate_504x504') 

# Define application constants and configurations.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = [
    "No apparent retinopathy", 
    "Mild non-proliferative retinopathy", 
    "Moderate NPDR", 
    "Severe NPDR", 
    "Proliferative diabetic retinopathy"
]
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
# Set a maximum file size for uploads (e.g., 16MB).
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure that the necessary directories exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if an uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handles the main page, which includes the file upload form (GET) and
    the file processing logic (POST).
    """
    # If the request is a POST, it means a file is being uploaded.
    if request.method == 'POST':
        # Basic validation to ensure a file was actually included in the request.
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # If the file is valid and has an allowed extension...
        if file and allowed_file(file.filename):
            # Use secure_filename to prevent directory traversal attacks.
            filename = secure_filename(file.filename)
            uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_image_path)

            # Call the main backend function to process the image.
            # This function runs the model, generates the attention map, and saves results.
            predicted_class_index, confidence, final_image_path, attention_data_path, lesion_data_path = generate_final_explanation(
                uploaded_image_path, MASKS_DIR, app.config['RESULTS_FOLDER']
            )

            # If processing was successful, render the results page.
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
            else:
                # Handle cases where the backend processing failed.
                return "Error processing image.", 500
        else:
            # Handle invalid file types.
            return "Invalid file type.", 400
            
    # For a GET request, just display the upload form.
    return render_template('upload.html')

@app.route('/annotate/<original_filename>/<attention_data_filename>/<lesion_data_filename>')
def annotate(original_filename, attention_data_filename, lesion_data_filename):
    """
    Prepares and serves the data needed for the interactive annotation page.
    This page uses JavaScript to render the attention map and lesion boxes
    over the original image on a canvas.
    """
    attention_data, lesion_data, image_data_url = [], [], None
    try:
        # 1. Read the original image and Base64-encode it.
        # This allows embedding the image directly into the HTML as a data URL,
        # which simplifies client-side rendering.
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        image_data_url = f"data:image/png;base64,{encoded_string}"
        
        # 2. Load the attention map and lesion data from their JSON files.
        with open(os.path.join(app.config['RESULTS_FOLDER'], attention_data_filename), 'r') as f:
            attention_data = json.load(f)
        with open(os.path.join(app.config['RESULTS_FOLDER'], lesion_data_filename), 'r') as f:
            lesion_data = json.load(f)
            
    except Exception as e:
        print(f"Error preparing data for annotation page: {e}")

    # Pass all the necessary data to the `annotate.html` template.
    return render_template('annotate.html',
                           original_image_data_url=image_data_url,
                           attention_data=attention_data,
                           lesion_data=lesion_data
                           )

@app.route('/uploads_display/<filename>')
def send_uploaded_file(filename):
    """
    A utility route to serve files directly from the UPLOAD_FOLDER.
    This is needed because Flask, by default, only serves files from the 'static'
    directory. This route makes the original uploaded images accessible to the browser.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Main Execution Block ---
if __name__ == '__main__':
    # Run the Flask app.
    # host='0.0.0.0' makes the app accessible on the local network.
    # debug=True enables auto-reloading and provides detailed error pages.
    # This should be set to False in a production environment.
    app.run(host='0.0.0.0', port=5000, debug=True)