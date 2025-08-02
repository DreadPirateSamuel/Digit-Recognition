# Digit Recognition

This project is a digit recognition application that allows users to draw single, double, or triple digits (0-9) on a canvas using Pygame and predict them using a convolutional neural network (CNN) trained on the MNIST dataset. The model achieves >98% accuracy on single-digit predictions, with robust multi-digit recognition using fixed segmentation.

## Features
- Single Mode: Draw one digit (0-9) on a 560x560 canvas, predicted with high accuracy.
- Double Mode: Draw two digits, recognized as a two-digit number (e.g., "88").
- Triple Mode: Draw three digits, recognized as a three-digit number (e.g., "888").
- User Interface: Pygame-based drawing canvas with mode toggling and guiding lines for multi-digit modes.
- Model: 3-layer CNN trained on MNIST with data augmentation for robustness.

## Requirements
- Python 3.13.5
- Dependencies (listed in `requirements.txt`):
  - `pygame==2.6.1`
  - `torch==2.7.1`
  - `torchvision==0.22.1`
  - `pillow==11.3.0`
  - `numpy==2.1.2`

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd digit-recognition
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**
   Run `train.py` to train the CNN on MNIST:
   ```bash
   python train.py
   ```
   - Downloads MNIST dataset.
   - Trains for 10 epochs (~10-20 minutes on CPU).
   - Saves model as `mnist_model.pth`.
   - Expected output:
     ```
     MNIST train labels: 0 to 9
     Epoch 1/10, Loss: <value>, Test Accuracy: <value>%
     ...
     Epoch 10/10, Loss: <value>, Test Accuracy: <value>%
     Model saved as mnist_model.pth
     ```
   - For faster training, install GPU support:
     ```bash
     pip install torch==2.7.1+cu121
     ```

5. **Run the Application**
   ```bash
   python main.py
   ```

## Usage
- Interface: A 560x560 Pygame window opens, titled "Draw a Digit."
- Modes (toggle with `M` key):
  - Single: Draw one digit in the center (e.g., "8").
  - Double: Draw two digits around x=100 and x=460 (gray line at x=280).
  - Triple: Draw three digits around x=70, 280, 490 (gray lines at x=186, 372).
  - Mode is displayed at the bottom.
- Drawing:
  - Click and drag to draw bold, clear digits.
  - Release to see prediction (e.g., `Predicted: 8`, `Predicted: 88`, `Predicted: 888`).
  - Draw within segment centers for multi-digit modes (e.g., double: "8" at x=100, "1" at x=460).
- Controls:
  - `C`: Clear canvas.
  - `Q`: Quit or close the window.
- Example Output (in terminal):
  ```
  Segment 0 mean pixel: 15.23, max pixel: 255.00, non-zero pixels: 200, x-range: 0-360
  Segment 0 predicted: 8 (confidence: 0.95, second: 0 with 0.03)
  Predicted: 8
  ```

## Troubleshooting
- Model Loading Error:
  - If `RuntimeError` occurs, retrain the model:
    ```bash
    del mnist_model.pth
    python train.py
    ```
- Inaccurate Predictions:
  - Ensure digits are bold and centered (e.g., double mode: x=100, 460).
  - Check terminal for segment stats (mean/max pixel, non-zero pixels, confidence).
  - If empty segments predict digits, share output to adjust thresholding.
- Single-Digit Issues:
  - If digits like "8" are mispredicted, share test accuracy from `train.py` and terminal output.
- Pygame Warning: The `pkg_resources` warning is harmless.
- VSCode: Select the virtual environment interpreter (`Ctrl+Shift+P` > "Python: Select Interpreter" > `.\venv\Scripts\python.exe`).

## Project Structure
- `train.py`: Trains the CNN on MNIST and saves `mnist_model.pth`.
- `main.py`: Runs the Pygame interface for drawing and predicting digits.
- `requirements.txt`: Lists dependencies.
- `mnist_model.pth`: Trained model file (generated after running `train.py`).

## Contributing
Feel free to submit issues or pull requests for improvements, such as enhanced preprocessing or additional features.
