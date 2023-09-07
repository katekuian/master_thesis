import numpy as np
import os

seed = 9876

image_folder = './WE3DS/images/'
annotation_folder = './WE3DS/annotations/'
annotations_aggregated_folder = './WE3DS/annotations_aggregated/'
# Define the paths to the images and annotations
all_image_names = np.array(os.listdir(image_folder))

plant_classification = {
    'void': 'void',
    'soil': 'soil',
    'broad bean': 'crop',
    'corn spurry': 'weed',
    'red-root amaranth': 'weed',
    'common buckwheat': 'crop',
    'pea': 'crop',
    'red fingergrass': 'weed',
    'common wild oat': 'weed',
    'cornflower': 'weed',
    'corn cockle': 'weed',
    'corn': 'crop',
    'milk thistle': 'weed',
    'rye brome': 'weed',
    'soybean': 'crop',
    'sunflower': 'crop',
    'narrow-leaved plantain': 'weed',
    'small-flower geranium': 'weed',
    'sugar beet': 'crop'
}

crop_indices = [index for index, value in enumerate(plant_classification) if plant_classification[value] == 'crop']
weed_indices = [index for index, value in enumerate(plant_classification) if plant_classification[value] == 'weed']

weed_plants = [plant_name.replace(" ", "_") for plant_name, classification in plant_classification.items() if classification == 'weed']