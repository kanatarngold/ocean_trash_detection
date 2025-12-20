# How to Train a Custom Ocean Trash Model

To detect trash specifically **in water** (underwater or surface), you should use a specialized dataset.

## Recommended Datasets (Roboflow Universe)

Since direct links can expire, please **search** for these exact names on [Roboflow Universe](https://universe.roboflow.com/):

### 1. For Surface Floating Trash (Best Option)
*   **Search Term:** `Ocean Plastics Waste Detection`
*   **Look for:** A project with ~2000+ images.
*   **Classes:** `Plastic Bottle`, `Plastic Bag`, `Can`, etc.

### 2. Alternative for Floating Waste
*   **Search Term:** `flow Object Detection Model`
*   **Look for:** A project by "school" or similar, with ~2000 images.
*   **Best for:** General floating debris in rivers/lakes.

### 3. For Underwater Trash
*   **Search Term:** `DeepTrash` or `Underwater Trash`
*   **Look for:** Projects based on JAMSTEC data.

---

## Training Instructions (Google Colab)

1.  **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com/)
2.  **Upload Notebook**: Upload the `train_custom_model.ipynb` file from your project folder.
3.  **Get Dataset Link**:
    *   Go to the Roboflow project page you found (e.g., "Ocean Plastics Waste Detection").
    *   Click **Download Dataset**.
    *   Select Format: **COCO**.
    *   Select **"Show Download Code"**.
    *   **Copy** the code snippet (it starts with `!curl ...`).
4.  **Paste Link in Notebook**:
    *   In the notebook, find **Step 2**.
    *   Paste the code snippet there.
5.  **Run All Cells**:
    *   The notebook will train the model and download `model.tflite`.
6.  **Install**:
    *   Copy `model.tflite` and `labels.txt` to `backend/models/`.
    *   Restart the app.
