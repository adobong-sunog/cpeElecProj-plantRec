import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Optional focal loss
try:
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy
    FOCAL_LOSS_AVAILABLE = True
except ImportError:
    FOCAL_LOSS_AVAILABLE = False

# Configuration
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f'training_results_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

img_width, img_height = 224, 224
batch_size = 32
initial_epochs = 10
fine_tune_epochs = 10
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

# Advanced Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
)
validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)
test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Save class indices
d = train_generator.class_indices
num_classes = len(d)
with open(os.path.join(output_dir, 'class_indices.json'), 'w') as f:
    json.dump(d, f)
print(f"Classes: {d}")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

# Build model
tf.keras.backend.clear_session()
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(img_width, img_height, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Use focal loss or sparse_categorical
loss_fn = (SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
           if FOCAL_LOSS_AVAILABLE else 'sparse_categorical_crossentropy')
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss=loss_fn,
    metrics=['accuracy']
)

# Callbacks for phase 1
checkpoint1 = ModelCheckpoint(
    os.path.join(output_dir, 'best_head.h5'),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
early_stop1 = EarlyStopping(
    monitor='val_loss', patience=4, restore_best_weights=True, verbose=1
)
reduce_lr1 = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=1
)

# Train head
hist1 = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=[checkpoint1, early_stop1, reduce_lr1]
)

# Fine-tune last layers
fine_tune_at = 100
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

checkpoint2 = ModelCheckpoint(
    os.path.join(output_dir, 'best_finetune.h5'),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
early_stop2 = EarlyStopping(
    monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1
)
reduce_lr2 = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=1, verbose=1
)

hist2 = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=[checkpoint2, early_stop2, reduce_lr2]
)

# Save final model
model.save(os.path.join(output_dir, 'plant_disease_model.h5'))
model.save('plant_disease_model.h5')

# Evaluation on test set
print("Evaluating on test data...")
test_loss, test_acc = model.evaluate(
    test_generator,
    steps=test_generator.samples // batch_size
)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Predictions & reports
preds = model.predict(test_generator, steps=test_generator.samples // batch_size)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes[:len(pred_classes)]
labels = list(d.keys())

# Classification report
report = classification_report(true_classes, pred_classes,
                               target_names=labels, output_dict=True)
with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(classification_report(true_classes, pred_classes, target_names=labels))

# Confusion matrix
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

# Save class accuracies
class_acc = {labels[i]: float(accuracy_score(
    true_classes == i, pred_classes[true_classes == i]
)) for i in range(num_classes)}
with open(os.path.join(output_dir, 'class_accuracy.json'), 'w') as f:
    json.dump(class_acc, f, indent=2)

print(f"Results saved in {output_dir}")
