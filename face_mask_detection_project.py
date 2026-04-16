!pip install kaggle tensorflow matplotlib numpy pillow requests -q

import os
import zipfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from io import BytesIO
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (AveragePooling2D, Flatten, Dense,
                                     Dropout, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from google.colab import files

print(f"TensorFlow version: {tf.__version__}")
print("All libraries imported successfully!")

def setup_kaggle_and_download():
    """Upload kaggle.json, configure credentials, and download the dataset."""

    print("📂 Please upload your kaggle.json file...")
    uploaded = files.upload()  # triggers file upload dialog in Colab

    # Move kaggle.json to the expected location
    os.makedirs("/root/.kaggle", exist_ok=True)
    shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("✅ kaggle.json configured successfully.")

    # Download dataset
    print("\n📥 Downloading face-mask-dataset from Kaggle...")
    os.system("kaggle datasets download -d omkargurav/face-mask-dataset -p /content/data")

    # Unzip dataset
    zip_path = "/content/data/face-mask-dataset.zip"
    extract_path = "/content/data/face-mask-dataset"

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"✅ Dataset extracted to: {extract_path}")
    return extract_path

DATASET_PATH = setup_kaggle_and_download()

def explore_dataset(dataset_path):
    """Print dataset folder structure and class counts."""

    print("📁 Dataset structure:")
    for root, dirs, files_list in os.walk(dataset_path):
        level = root.replace(dataset_path, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level == 1:  # show file counts at class level
            sub_indent = "  " * (level + 1)
            print(f"{sub_indent}→ {len(files_list)} images")

    # Find the data directory containing 'with_mask' and 'without_mask'
    for root, dirs, _ in os.walk(dataset_path):
        if "with_mask" in dirs and "without_mask" in dirs:
            print(f"\n✅ Found class folders at: {root}")
            return root

    raise FileNotFoundError("Could not find 'with_mask' / 'without_mask' folders.")

DATA_DIR = explore_dataset(DATASET_PATH)

# ── Image settings ──
IMG_SIZE    = 224        # MobileNetV2 expects 224×224
IMG_SHAPE   = (IMG_SIZE, IMG_SIZE, 3)

# ── Training settings ──
BATCH_SIZE  = 32
EPOCHS      = 20
LEARNING_RATE = 1e-4

# ── Data split ──
VALIDATION_SPLIT = 0.20  # 80% train, 20% validation

# ── Class labels ──
CLASSES = ["with_mask", "without_mask"]
CLASS_LABELS = {0: "Mask ✅", 1: "No Mask ❌"}

print("⚙️  Configuration loaded:")
print(f"   Image size    : {IMG_SIZE}×{IMG_SIZE}")
print(f"   Batch size    : {BATCH_SIZE}")
print(f"   Epochs        : {EPOCHS}")
print(f"   Learning rate : {LEARNING_RATE}")
print(f"   Val split     : {VALIDATION_SPLIT * 100:.0f}%")

def build_data_generators(data_dir):
    """
    Create ImageDataGenerator for training (with augmentation)
    and validation (no augmentation, only rescaling).
    """

    # Training generator — augmentation applied
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # MobileNetV2 preprocessing
        rotation_range=20,            # rotate images up to 20°
        zoom_range=0.15,              # random zoom
        width_shift_range=0.2,        # horizontal shift
        height_shift_range=0.2,       # vertical shift
        shear_range=0.15,             # shear transformation
        horizontal_flip=True,         # random horizontal flip
        fill_mode="nearest",          # fill strategy for empty pixels
        validation_split=VALIDATION_SPLIT
    )

    # Validation generator — only preprocessing, no augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT
    )

    # Training data flow
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42
    )

    # Validation data flow
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42
    )

    print(f"\n✅ Generators created:")
    print(f"   Training samples   : {train_generator.samples}")
    print(f"   Validation samples : {val_generator.samples}")
    print(f"   Class indices      : {train_generator.class_indices}")

    return train_generator, val_generator

train_gen, val_gen = build_data_generators(DATA_DIR)

def visualise_samples(generator, num_images=10):
    """Display a grid of sample images with their class labels."""

    images, labels = next(generator)
    class_names = {v: k for k, v in generator.class_indices.items()}

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("Sample Training Images (after augmentation)", fontsize=14)

    for i, ax in enumerate(axes.flatten()):
        if i >= num_images:
            break
        # Reverse MobileNetV2 preprocessing for display
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())  # normalise to [0,1]
        label_idx = np.argmax(labels[i])
        label_name = class_names[label_idx]
        color = "green" if label_name == "with_mask" else "red"
        ax.imshow(img)
        ax.set_title(label_name, color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("/content/sample_images.png", dpi=100)
    plt.show()
    print("✅ Sample images saved to /content/sample_images.png")

visualise_samples(train_gen)

def build_model(img_shape=IMG_SHAPE, num_classes=2):
    """
    Construct the transfer learning model:
      Base  : MobileNetV2 (pretrained on ImageNet, frozen)
      Head  : AveragePooling2D → Flatten → Dense(128) → Dropout → Dense(2, softmax)
    """

    # ── Base model: MobileNetV2 ──
    # include_top=False excludes the original ImageNet classifier head
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=img_shape)
    )

    # Freeze all base model layers — we only train the custom head initially
    base_model.trainable = False

    # ── Custom classification head ──
    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)   # spatial pooling
    head = Flatten()(head)                             # flatten to 1D
    head = Dense(128, activation="relu")(head)         # fully connected layer
    head = Dropout(0.5)(head)                          # regularisation
    head = Dense(num_classes, activation="softmax")(head)  # output layer

    # ── Combine base + head ──
    model = Model(inputs=base_model.input, outputs=head)

    print("✅ Model built successfully.")
    print(f"   Total layers    : {len(model.layers)}")
    print(f"   Trainable params: {model.count_params():,}")

    return model

model = build_model()
model.summary()

def compile_model(model, learning_rate=LEARNING_RATE):
    """Compile the model with Adam optimiser and binary crossentropy loss."""

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",   # suitable for 2-class softmax
        metrics=["accuracy"]
    )
    print(f"✅ Model compiled with Adam (lr={learning_rate}) and binary_crossentropy.")
    return model

model = compile_model(model)

def get_callbacks():
    """
    Define training callbacks:
    - EarlyStopping  : stop if val_loss doesn't improve for 5 epochs
    - ReduceLROnPlateau : halve learning rate if val_loss plateaus for 3 epochs
    """

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    return [early_stop, reduce_lr]

callbacks = get_callbacks()
print("✅ Callbacks configured: EarlyStopping + ReduceLROnPlateau")

def train_model(model, train_gen, val_gen, epochs=EPOCHS, callbacks=None):
    """Train the model and return the history object."""

    print(f"\n🚀 Starting training for up to {epochs} epochs...")
    print("   (EarlyStopping will halt early if val_loss stops improving)\n")

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks
    )

    print("\n✅ Training complete!")
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]
    print(f"   Final train accuracy : {final_train_acc:.4f}")
    print(f"   Final val accuracy   : {final_val_acc:.4f}")

    return history

history = train_model(model, train_gen, val_gen, EPOCHS, callbacks)

def plot_training_history(history):
    """
    Plot training & validation loss and accuracy side by side.
    Saved to /content/training_curves.png
    """

    epochs_ran = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=15)

    # ── Loss plot ──
    ax1.plot(epochs_ran, history.history["loss"],     "b-o", label="Train Loss",      linewidth=2)
    ax1.plot(epochs_ran, history.history["val_loss"], "r-o", label="Validation Loss", linewidth=2)
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy plot ──
    ax2.plot(epochs_ran, history.history["accuracy"],     "b-o", label="Train Accuracy",      linewidth=2)
    ax2.plot(epochs_ran, history.history["val_accuracy"], "r-o", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/content/training_curves.png", dpi=150)
    plt.show()
    print("✅ Training curves saved to /content/training_curves.png")

plot_training_history(history)

def evaluate_model(model, val_gen):
    """Compute final loss and accuracy on the validation set."""

    print("📊 Evaluating model on validation set...")
    val_gen.reset()  # reset to start from the beginning
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    print(f"\n✅ Validation Results:")
    print(f"   Loss     : {loss:.4f}")
    print(f"   Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    return loss, accuracy

val_loss, val_acc = evaluate_model(model, val_gen)

MODEL_SAVE_PATH = "/content/face_mask_detector.keras"

model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Resize a PIL Image to 224×224, apply MobileNetV2 preprocessing,
    and expand dims for batch inference.
    """
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)       # scale to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0) # shape: (1, 224, 224, 3)
    return img_array


def predict_mask(img: Image.Image, model: Model) -> tuple:
    """
    Run inference on a PIL Image.

    Returns:
        label      (str)   : 'Mask ✅' or 'No Mask ❌'
        confidence (float) : prediction confidence in [0, 1]
        class_idx  (int)   : 0 = with_mask, 1 = without_mask
    """
    processed  = preprocess_image(img)
    predictions = model.predict(processed, verbose=0)[0]  # shape: (2,)
    class_idx   = int(np.argmax(predictions))
    confidence  = float(predictions[class_idx])
    label       = CLASS_LABELS[class_idx]
    return label, confidence, class_idx


def display_prediction(img: Image.Image, label: str,
                       confidence: float, class_idx: int,
                       source: str = ""):
    """Display the image with the prediction label overlaid."""

    color = "green" if class_idx == 0 else "red"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.set_title(
        f"{label}\nConfidence: {confidence * 100:.1f}%"
        + (f"\nSource: {source}" if source else ""),
        fontsize=13,
        color=color,
        fontweight="bold"
    )
    ax.axis("off")

    # Coloured border around the image
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4)

    plt.tight_layout()
    plt.savefig("/content/prediction_result.png", dpi=150)
    plt.show()
    print(f"✅ Prediction: {label}  |  Confidence: {confidence * 100:.1f}%")

def predict_from_url(image_url: str, model: Model):
    """
    Download an image from a URL and run mask detection.

    Usage:
        predict_from_url("https://example.com/person.jpg", model)
    """
    print(f"🌐 Fetching image from URL...")
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to load image from URL: {e}")
        return

    label, confidence, class_idx = predict_mask(img, model)
    display_prediction(img, label, confidence, class_idx,
                       source=image_url[:60] + "...")


# ── Example usage (replace with any public image URL) ──
SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Face_mask_China_2020.jpg/320px-Face_mask_China_2020.jpg"
predict_from_url(SAMPLE_URL, model)

def predict_from_upload(model: Model):
    """
    Prompt the user to upload an image file from their local machine
    and run mask detection on it.

    Usage:
        predict_from_upload(model)
    """
    print("📤 Please upload an image file (jpg, jpeg, png)...")
    uploaded = files.upload()

    for filename, content in uploaded.items():
        print(f"\n🔍 Processing: {filename}")
        img = Image.open(BytesIO(content)).convert("RGB")
        label, confidence, class_idx = predict_mask(img, model)
        display_prediction(img, label, confidence, class_idx,
                           source=filename)


# ── Uncomment the line below to test with an uploaded image ──
predict_from_upload(model)

def show_batch_predictions(model, val_gen, num_images=8):
    """
    Run inference on a batch of validation images and display
    results in a grid showing correct vs incorrect predictions.
    """

    val_gen.reset()
    images, true_labels = next(val_gen)
    class_names = {v: k for k, v in val_gen.class_indices.items()}

    preds = model.predict(images[:num_images], verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    true_indices = np.argmax(true_labels[:num_images], axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Batch Predictions — Green border = Correct, Red border = Wrong",
                 fontsize=13)

    for i, ax in enumerate(axes.flatten()):
        if i >= num_images:
            break

        # Reverse preprocessing for display
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())

        pred_name = class_names[pred_indices[i]]
        true_name = class_names[true_indices[i]]
        confidence = preds[i][pred_indices[i]]
        is_correct = pred_indices[i] == true_indices[i]

        border_color = "green" if is_correct else "red"
        status = "✅" if is_correct else "❌"

        ax.imshow(img)
        ax.set_title(
            f"Pred: {pred_name}\nTrue: {true_name} ({confidence:.0%}) {status}",
            fontsize=9,
            color=border_color
        )
        ax.axis("off")

        # Draw coloured border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig("/content/batch_predictions.png", dpi=150)
    plt.show()
    print("✅ Batch predictions saved to /content/batch_predictions.png")

show_batch_predictions(model, val_gen)

from google.colab import files

# This will download the file directly to your 'Downloads' folder
files.download('/content/face_mask_detector.keras')
print("Check your browser downloads for the model file!")

print("\n" + "=" * 55)
print("  FACE MASK DETECTION — PROJECT COMPLETE")
print("=" * 55)
print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  Model saved to      : {MODEL_SAVE_PATH}")
print()
print("  Output files in /content/:")
print("    • face_mask_detector.keras  — trained model")
print("    • training_curves.png       — loss & accuracy plots")
print("    • prediction_result.png     — URL inference result")
print("    • batch_predictions.png     — batch inference grid")
print("    • sample_images.png         — augmented samples")
