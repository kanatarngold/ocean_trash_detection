# How to Train a Custom Ocean Trash Model

To detect trash specifically **in water** (underwater or surface), you should use a specialized dataset.

## Recommended Datasets (Roboflow Universe)

Since direct links can expire, please **search** for these exact names on [Roboflow Universe](https://universe.roboflow.com/):

### 1. Ocean Plastics Waste Detection (Highly Recommended)
*   **Search Term:** `Ocean Plastics Waste Detection`
*   **User/Workspace:** `abdelaadimKHRISS` or `khrissAbdelaadim`
*   **Content:** ~2000+ images of floating plastic bottles, bags, and debris.
*   **Why:** Specifically trained on water backgrounds.

### 2. Marine Litter 2018-2019
*   **Search Term:** `Marine litter 2018-2019`
*   **Content:** ~5000 images! (Bottles, Hardplastic, Softplastic).
*   **Why:** Very large dataset, good for variety.

### Raspberry Pi Optimization (The "Sweet Spot")

We are now using **YOLOv8 Small** + **640px Resolution** + **Int8 Quantization**.

### Why this is better for Pi:
*   **Speed**: ~4-6 FPS (vs <1 FPS with Large).
*   **Accuracy**: "Small" is much smarter than "Nano".
*   **Quantization**: Makes the model 4x smaller and faster without losing much accuracy.

### Hardware Upgrade (Optional)
If you need **30+ FPS**, buy a **Google Coral USB Accelerator** (~$60).
*   It plugs into the Pi USB port.
*   It accelerates TFLite models massively.

---

## State of the Art Training (Maximum Accuracy)

We are now using **YOLOv8 Large** + **1280px Resolution**.

### How to Merge Datasets (Secret Weapon)
To get the best results, combine multiple datasets on Roboflow:

1.  **Create a New Project** on Roboflow (e.g. "Mega-Trash-Dataset").
2.  **Find Datasets** on Roboflow Universe (TACO, Ocean Plastics, etc.).
3.  Click **"Add to Project"** on each dataset page.
4.  Select your new "Mega-Trash-Dataset".
5.  **Generate Version**:
    *   Preprocessing: **Auto-Orient**, **Resize -> Stretch to 1280x1280**.
    *   Augmentation: **Flip**, **Rotation**, **Blur**, **Noise** (adds variety).
6.  **Get the Link**: Use this new link in the notebook!

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
