import os
import numpy as np
import h5py
from tensorflow.keras.models import save_model

# Directory to save the weights
weights_directory = 'osic_model_weights'

# Ensure the directory exists
os.makedirs(weights_directory, exist_ok=True)

# Function to generate random weights
def generate_weights(model, model_class):
    # Generate random weights
    weights = [np.random.randn(*w.shape) for w in model.get_weights()]
    
    # Save weights to HDF5 file
    with h5py.File(os.path.join(weights_directory, f'{model_class}.h5'), 'w') as f:
        for i, w in enumerate(weights):
            f.create_dataset(f'weight_{i}', data=w)

# Models and their corresponding classes
model_classes = ['b5'] #['b0','b1','b2','b3',b4','b5','b6','b7']

# Assuming models is a list containing your models
# Initialize your models here
models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]

# Generate weights for each model
for model, model_class in zip(models, model_classes):
    generate_weights(model, model_class)

print("Weights generated successfully.")
