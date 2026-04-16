
## Quick Start

### 1. Train the model
Open `face_mask_detection_project.py` in Colab and run all cells.

### 2. Run webcam detection
```bash
pip install -r requirements.txt
python detect_mask.py
```

Press `q` to quit.

## Results

Model accuracy: ~95% on test set (your actual result from notebook).

## Screenshots

### Training Curve
![Training](outputs/training_curve.png)

### Webcam Demo
![Webcam](outputs/webcam_demo.png)

## Requirements

```bash
pip install -r requirements.txt
```

## Model Details

- **Architecture:** MobileNetV2 + classification head.
- **Input:** 224x224 RGB images.
- **Classes:** `with_mask`, `without_mask`.

## Future Work

- Add `mask_weared_incorrect` class.
- Deploy as Streamlit app.
- Improve face detection speed.
- Mobile deployment.

## License

MIT License.
