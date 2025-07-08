# Vision Transformer Explainability for Diabetic Retinopathy

This project provides a web-based tool to visualize and understand the decisions of a Vision Transformer (ViT) model trained to classify diabetic retinopathy (DR) from retinal fundus images. 
It uses the "attention rollout" method to generate saliency maps, highlighting the image regions the model focuses on for its predictions.

---

## Key Features

-   **DR Classification**: Classifies retinal images into one of five stages of diabetic retinopathy using a fine-tuned DINOv2 (ViT-G/14) model.
-   **Explainable AI (XAI)**: Generates attention maps to provide visual explanations for the model's predictions.
-   **Lesion Hit-Analysis**: Automatically identifies ground-truth lesion locations (from corresponding mask images) and determines if the model's attention is focused on them. Bounding boxes are color-coded as a "hit" (green) or "miss" (red) based on attention coverage.
-   **Interactive Web Interface**: A Flask-based web application allows users to upload an image and view the classification result alongside the visual explanation.
-   **Detailed Annotation View**: An interactive page that overlays the attention heatmap and lesion data on the original image for in-depth analysis.

---

## How It Works

The workflow is orchestrated by a Flask web server (`app.py`) that handles user interactions and calls the core AI logic from `attention_explainer.py`.

1.  **Image Upload**: A user uploads a retinal fundus image through the web interface.
2.  **Preprocessing**: The image is preprocessed with a specific pipeline and saved in to 504x504 pixels.
3.  **Model Inference**: The DINOv2 model with a custom classification head processes the image to predict the DR stage and confidence score.
4.  **Attention Extraction**: We use forward hooks on the model's attention layers to capture the attention weights during the forward pass. The "attention rollout" algorithm is then applied to compose these weights into a single, comprehensive saliency map.
5.  **Visualization Generation**:
    * The attention map is visualized as a JET colormap heatmap.
    * This heatmap is overlaid on the original image to create a preview of the model's focus.
    * The system searches for a corresponding lesion mask image based on the input filename.
    * If a mask is found, bounding boxes for each lesion are extracted.
    * Each bounding box is analyzed to calculate the percentage of "hot" attention pixels it contains. If the ratio exceeds a threshold (20%), the box is colored green (hit); otherwise, it's colored red (miss).
6.  **Display Results**: The application displays the predicted class, confidence score, and the generated analysis image with the overlaid heatmap and bounding boxes.

---

## Usage

1.  **Run the Flask Application**
    ```bash
    python app.py
    ```

2.  **Access the Web Interface**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

3.  **Upload an Image**
    Click "Choose File", select a retinal fundus image (`.png`, `.jpg`, `.jpeg`), and click "Upload".

4.  **View the Results**
    The application will process the image and redirect you to the results page, which shows:
    -   The predicted DR class and confidence score.
    -   The analysis image with the attention heatmap and color-coded lesion bounding boxes.
    -   A link to the interactive annotation view.

---

The attention map generation is based on the paper [Vision Transformer-based explainability for computer-aided diagnosis](https://www.nature.com/articles/s41598-023-45532-z) and the original "attention rollout" method proposed by [Abnar & Zuidema (2020)](https://arxiv.org/abs/2005.00928).
