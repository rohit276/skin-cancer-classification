import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import cv2
import shutil
from datetime import datetime
from tqdm import tqdm

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  
NUM_CLASSES = 7
DATASET_PATH = 'dataset/HAM10000'
USE_METADATA = False
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('runs', exist_ok=True)

# Create a timestamp for this training run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/{TIMESTAMP}"
os.makedirs(RUN_DIR, exist_ok=True)

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{RUN_DIR}/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_metadata():
    """Load and preprocess the metadata."""
    logger.info("Loading metadata...")
    
    # Read metadata
    metadata_path = f'{DATASET_PATH}/HAM10000_metadata.csv'
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found at {metadata_path}")
        return None
        
    df = pd.read_csv(metadata_path)
    
    # Map diagnosis to numerical labels
    diagnosis_mapping = {
        'akiec': 0,  # Actinic Keratoses
        'bcc': 1,    # Basal Cell Carcinoma
        'bkl': 2,    # Benign Keratosis
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic Nevi
        'vasc': 6    # Vascular Lesions
    }
    
    # Store numerical labels
    df['label_num'] = df['dx'].map(diagnosis_mapping)
    # Use dx as the label column
    df['label'] = df['dx']
    
    # Create path column
    df['path'] = df['image_id'].apply(lambda x: f'{DATASET_PATH}/images/{x}.jpg')
    
    # Check if files exist
    existing_files = df['path'].apply(os.path.exists)
    if not existing_files.all():
        logger.warning(f"Found {(~existing_files).sum()} missing image files")
        df = df[existing_files]
    
    # Check class distribution
    class_counts = df['label'].value_counts()
    logger.info("Class distribution:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count} ({count/len(df)*100:.2f}%)")
    
    return df, diagnosis_mapping

def prepare_data(df):
    """Prepare the dataset for training."""
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=(VALIDATION_SPLIT + TEST_SPLIT), random_state=SEED, stratify=df['label_num']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_SPLIT/(VALIDATION_SPLIT + TEST_SPLIT), 
        random_state=SEED, stratify=temp_df['label_num']
    )
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Testing samples: {len(test_df)}")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(x='dx', data=df)
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{RUN_DIR}/class_distribution.png')
    
    return train_df, val_df, test_df

def preprocess_image(img_path, is_training=False):
    """Preprocess an image for model input."""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Failed to read image: {img_path}")
        # Return a blank image as fallback
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Apply enhanced augmentations if training
    if is_training:
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 0)
        
        # Random rotation (more angles)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-45, 45)  # Increased angle range
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
        
        # Random brightness/contrast
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.7, 1.3)  # Wider contrast range
            beta = np.random.uniform(-20, 20)    # Wider brightness range
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
        # Random zoom
        if np.random.random() > 0.7:  # 30% chance
            zoom = np.random.uniform(0.8, 1.2)
            h, w = img.shape[:2]
            crop_h, crop_w = int(h * zoom), int(w * zoom)
            
            # Ensure crop dimensions are valid
            if crop_h < h and crop_w < w:
                # Random crop position
                start_h = np.random.randint(0, h - crop_h)
                start_w = np.random.randint(0, w - crop_w)
                img = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
        # Random blur
        if np.random.random() > 0.8:  # 20% chance
            kernel_size = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
        # Random color jitter
        if np.random.random() > 0.7:  # 30% chance
            # Randomly adjust hue, saturation, value
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(img_hsv)
            
            # Adjust hue
            h = h.astype(np.float32)
            h += np.random.uniform(-10, 10)
            h = np.clip(h, 0, 179).astype(np.uint8)
            
            # Adjust saturation
            s = s.astype(np.float32)
            s *= np.random.uniform(0.7, 1.3)
            s = np.clip(s, 0, 255).astype(np.uint8)
            
            # Adjust value
            v = v.astype(np.float32)
            v *= np.random.uniform(0.7, 1.3)
            v = np.clip(v, 0, 255).astype(np.uint8)
            
            img_hsv = cv2.merge([h, s, v])
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    return img

def create_tf_dataset(df, batch_size=BATCH_SIZE, is_training=False):
    """Create a TensorFlow dataset from a dataframe."""
    # Balance the dataset for training
    if is_training:
        # Get counts for each class
        class_counts = df['label_num'].value_counts()
        max_size = class_counts.max()
        
        # Create balanced dataframe by oversampling minority classes
        balanced_dfs = []
        for label, group in df.groupby('label_num'):
            if len(group) < max_size:
                # Oversample with replacement
                resampled = group.sample(n=max_size, replace=True, random_state=SEED)
                balanced_dfs.append(resampled)
            else:
                balanced_dfs.append(group)
        
        df = pd.concat(balanced_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
        logger.info(f"Balanced dataset size: {len(df)}")
    
    # Create a dataset of image paths and labels
    image_paths = df['path'].values
    labels = tf.keras.utils.to_categorical(df['label_num'].values, num_classes=NUM_CLASSES)
    
    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Map the preprocessing function
    dataset = dataset.map(
        lambda x, y: (tf.py_function(
            func=lambda path: preprocess_image(path.numpy().decode(), is_training),
            inp=[x],
            Tout=tf.float32
        ), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set shapes explicitly
    dataset = dataset.map(
        lambda x, y: (tf.ensure_shape(x, (IMG_SIZE, IMG_SIZE, 3)), y)
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_model():
    """Build an improved model for skin cancer classification."""
    # Input layer
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Base model - upgraded to EfficientNetB4 for better performance
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:  # Freeze fewer layers for better fine-tuning
        layer.trainable = False
    
    # Add improved classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # First dense block
    x = Dense(1024, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Second dense block
    x = Dense(512, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Third dense block for better feature extraction
    x = Dense(256, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(train_df, val_df):
    """Train the model with improved callbacks and monitoring."""
    # Create datasets
    train_dataset = create_tf_dataset(train_df, batch_size=BATCH_SIZE, is_training=True)
    val_dataset = create_tf_dataset(val_df, batch_size=BATCH_SIZE, is_training=False)
    
    # Build model
    model = build_model()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        f'{RUN_DIR}/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,  # Increased patience
        min_lr=1e-6,
        verbose=1
    )
    
    # Add TensorBoard callback for better monitoring
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f'{RUN_DIR}/logs',
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Train model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{RUN_DIR}/training_history.png')
    
    # Load the best model
    model = load_model(f'{RUN_DIR}/best_model.h5')
    
    return model

def evaluate_model(model, test_df, diagnosis_mapping):
    """Evaluate the model on the test set."""
    # Create test dataset
    test_dataset = create_tf_dataset(test_df, batch_size=BATCH_SIZE, is_training=False)
    
    # Get class names
    class_names = [k for k, v in sorted(diagnosis_mapping.items(), key=lambda item: item[1])]
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Predict on test data
    y_pred_probs = model.predict(test_dataset)
    y_true = tf.keras.utils.to_categorical(test_df['label_num'].values, num_classes=NUM_CLASSES)
    
    # Get predicted classes
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true_classes, y_pred)
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    report = classification_report(y_true_classes, y_pred, target_names=class_names, digits=4)
    logger.info(f"\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true_classes, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{RUN_DIR}/confusion_matrix.png')
    
    # Save the final model
    try:
        # Try to save in SavedModel format (more reliable)
        tf.keras.models.save_model(
            model, 
            'models/skin_cancer_model', 
            save_format='tf',
            include_optimizer=False
        )
        logger.info("Model saved as 'models/skin_cancer_model' in SavedModel format")
        
        # Also save to the run directory
        tf.keras.models.save_model(
            model, 
            f'{RUN_DIR}/final_model', 
            save_format='tf',
            include_optimizer=False
        )
        
        # Also save in H5 format for compatibility
        model.save('models/skin_cancer_model.h5')
        logger.info("Model also saved as 'models/skin_cancer_model.h5'")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        logger.info("Using best model from checkpoints instead")
        # Copy the best checkpoint model to the models directory
        shutil.copy(f'{RUN_DIR}/best_model.h5', 'models/skin_cancer_model.h5')
        logger.info("Best checkpoint model copied to 'models/skin_cancer_model.h5'")
    
    # Save model summary
    with open(f'{RUN_DIR}/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save test results
    test_results = {
        'accuracy': test_acc,
        'balanced_accuracy': balanced_acc,
        'classification_report': report
    }
    
    with open(f'{RUN_DIR}/test_results.txt', 'w') as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n\n")

def main():
    """Main function to run the training pipeline."""
    logger.info("Starting skin cancer classification model training")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset path {DATASET_PATH} does not exist")
        return
    
    # Load metadata
    df, diagnosis_mapping = load_metadata()
    if df is None:
        return
    
    # Prepare data
    train_df, val_df, test_df = prepare_data(df)
    
    # Train model
    model = train_model(train_df, val_df)
    
    # Evaluate model
    evaluate_model(model, test_df, diagnosis_mapping)
    
    logger.info("Training complete! The model is saved as 'models/skin_cancer_model.h5'")
    logger.info(f"All training artifacts are saved in '{RUN_DIR}'")

if __name__ == "__main__":
    main()
