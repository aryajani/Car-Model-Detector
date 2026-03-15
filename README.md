# Car Model Detector

**CSC273P Final Project**

An end-to-end deep learning system for automated car model recognition from images. This project combines transfer learning with EfficientNet-B0 to classify 1,085 distinct car make-model combinations, featuring both a backend inference API and an interactive React frontend.

---

## Project Overview

The Car Model Detector addresses the challenge of automatically identifying vehicle models from digital images. The system achieves 83.11% validation accuracy on a 1,085-class classification task using transfer learning with EfficientNet-B0 pre-trained on ImageNet.

### Key Features
- **Multi-class Classification**: 1,085 car make-model classes from the VMMRdb dataset
- **Transfer Learning**: Pre-trained EfficientNet-B0 backbone fine-tuned for car classification
- **Web Interface**: React-based frontend for easy image upload and results visualization
- **LLM Integration**: Groq API integration for generating detailed car information
- **High Performance**: 83.11% validation accuracy with class-balanced training

### Architecture
- **Frontend**: React application for image upload and results display
- **Backend**: FastAPI server for model inference and car information retrieval
- **ML Model**: EfficientNet-B0 with transfer learning (1,085 output classes)

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+ and npm
- CUDA-capable GPU (recommended for training)
- Git

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Car-Model-Detector
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Configure API Keys**
   - Get a GROQ API Key from [https://console.groq.com](https://console.groq.com)
   - Create a `.env` file in the `backend` folder:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

6. **Prepare Dataset**
   - Download the VMMRdb dataset
   - Place it in the parent directory: `../VMMRdb_make_model/VMMRdb_make_model`
   - The directory structure should have subdirectories for each car class

---

## Required Dependencies

### Backend Dependencies
```
torch==2.0.0+cu118
torchvision==0.15.0+cu118
fastapi==0.109.0
uvicorn==0.27.0
pillow==10.1.0
pydantic==2.5.2
python-dotenv==1.0.0
groq==0.4.0
scikit-learn==1.3.2
numpy==1.24.3
```

### Frontend Dependencies
```
react==19.2.4
react-dom==19.2.4
react-scripts==5.0.1
```

### Installation
For the backend:
```bash
cd backend
pip install -r requirements.txt
cd ..
```

For the frontend:
```bash
cd frontend
npm install
cd ..
```

---

## How to Train the Model

### Training a New Model from Scratch

The project includes three training approaches with different data splitting and loss strategies:

#### 1. Basic Training (random split, standard cross-entropy loss)
```bash
cd backend
python train.py
```

This trains an EfficientNet-B0 model with:
- Random 80/20 train/validation split
- Cross-entropy loss with label smoothing (α=0.1)
- Adam optimizer (lr=1×10⁻⁴)
- Batch size: 128 (train), 32 (validation)
- 15 epochs
- Mixed precision training for efficiency

**Output**: `efficientnet_checkpoint.pth`

#### 2. Class-Balanced Loss Training (stratified split, weighted loss)
```bash
python train_cbloss.py
```

This is the **recommended approach** that addresses class imbalance:
- Stratified 80/20 split preserving class distribution
- Class-balanced loss with β=0.999
- Cross-entropy loss with label smoothing (α=0.1)
- Adam optimizer (lr=1×10⁻⁴)
- Batch size: 32 (both train and validation)
- 15 epochs
- Logs metrics per epoch to `cbloss_train_output.txt`

**Output**: `cbloss_enet.pth` (best performing model)

#### 3. Random Split Alternative Training
```bash
python train_random.py
```

This serves as a comparison baseline:
- Random 80/20 split
- Standard cross-entropy loss with label smoothing
- Batch size: 32
- 15 epochs
- Logs to `random_train_output.txt`

**Output**: `random_enet.pth`

### Training Process Details
- Models use pre-trained ImageNet1K weights as initialization
- Only the final classification layer (1,085 classes) is modified
- Mixed precision training (`torch.amp`) speeds up computation
- All random seeds (Python, NumPy, PyTorch) set to 42 for reproducibility
- Training progress is printed per epoch showing loss values

---

## How to Evaluate the Model

### Evaluating a Trained Model

#### 1. Evaluate Class-Balanced Model (recommended)
```bash
python eval_cbloss.py
```

**Output**:
```
Total Accuracy: XX.XX%
<macro_f1_score>
```

The evaluation script uses stratified splitting to match the training procedure and reports both accuracy and macro F1-score.

#### 2. Evaluate Random Split Model
```bash
python eval_random.py
```

**Output**:
```
Total Accuracy: XX.XX%
<macro_f1_score>
```

#### 3. Basic Evaluation
```bash
python backend/eval.py
```

**Output**:
```
<accuracy>
```

Reports simple accuracy on the validation set.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions (primary metric)
- **Macro F1-Score**: Unweighted average of per-class F1 scores, accounting for class imbalance

---

## Running the Full Application

### Development Mode

1. **Terminal 1 - Start Backend**
   ```bash
   cd backend
   uvicorn app:app --reload
   ```
   The server will run at `http://localhost:8000`
   API docs available at `http://localhost:8000/docs`

2. **Terminal 2 - Start Frontend**
   ```bash
   cd frontend
   npm start
   ```
   The application will open at `http://localhost:3000`

### Using the Application
1. Upload a car image through the web interface
2. The model identifies the car make/model
3. Click to view detailed information generated by the LLM

---

## Expected Outputs

### Training Outputs

**Class-Balanced Training** (best approach):
- **Training Accuracy**: ~88.1% by epoch 15
- **Validation Accuracy**: ~83.1% by epoch 15
- **Training Loss**: Decreases from ~22.4 to ~18.8
- **Validation Loss**: Decreases from ~21.0 to ~19.6
- **File**: `cbloss_train_output.txt` contains per-epoch metrics in format:
  ```
  train_accuracy, train_loss, val_accuracy, val_loss
  ```

### Evaluation Outputs

**Per-Model Accuracy**:
- Class-Balanced approach: 83.1% ± variance
- Random split approach: ~78-82%
- Macro F1-Score: Available from evaluation scripts

### Inference Outputs

**API Response** (`/predict` endpoint):
```json
{
  "class_id": 42,
  "class_name": "Toyota Camry 2020"
}
```

**Car Info Response** (`/car-info` endpoint):
```json
{
  "info": "Toyota Camry is a mid-size sedan manufactured since 1982. Known for reliability and comfort, it offers multiple engine options including hybrid variants. Notable features include advanced safety systems, comfortable interior, and strong resale value."
}
```

---

## Model Performance Summary

| Configuration | Train Acc | Val Acc | Macro F1 | Best For |
|---------------|-----------|---------|----------|----------|
| Class-Balanced Loss | 88.12% | 83.11% | High | **Production Use** |
| Random Split | ~85% | ~78-80% | - | Baseline Comparison |

The class-balanced approach is recommended for production as it explicitly handles class imbalance in the dataset.

---

## Troubleshooting

### Common Issues

**CUDA/GPU not found**
- Ensure CUDA toolkit is installed
- Install correct PyTorch version: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

**API key error**
- Verify `.env` file is in `backend/` folder
- Check `GROQ_API_KEY` environment variable is set correctly

**Dataset not found**
- Ensure VMMRdb is in parent directory: `../VMMRdb_make_model/`
- Check directory structure has car model subdirectories

**Port already in use**
- Change port: `uvicorn app:app --reload --port 8001`

---

## References

- **EfficientNet**: Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- **Class-Balanced Learning**: Cui et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples"
- **VMMRdb Dataset**: Used for training and evaluation
- **Groq API**: For LLM-based car information generation

---

## Project Structure

```
Car-Model-Detector/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── train.py               # Basic training script
│   ├── eval.py                # Basic evaluation
│   ├── load_model.py          # Model loading utility
│   ├── predict_car.py         # Inference function
│   ├── chatbot.py             # Groq LLM integration
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── src/                   # React component source
│   ├── public/                # Static assets
│   ├── package.json           # Node dependencies
│   └── package-lock.json
├── train_cbloss.py            # Class-balanced training (recommended)
├── eval_cbloss.py             # Class-balanced evaluation
├── train_random.py            # Random split training
├── eval_random.py             # Random split evaluation
├── README.md                  # This file
└── cbloss_train_output.txt    # Training metrics log
```

---

**Last Updated**: March 2026
