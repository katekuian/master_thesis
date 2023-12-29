import numpy as np
import os

seed = 9876

image_folder = '../WE3DS/images/'
annotation_folder = '../WE3DS/annotations/'
annotations_binary_folder = '../WE3DS/annotations_binary/'
annotations_multiclass_folder = '../WE3DS/annotations_multiclass/'
image_filenames = np.array(os.listdir(image_folder))

meta_folder = '../meta/'
images_to_exclude_from_models_filepath = meta_folder + 'images_to_exclude_from_models.json'
weed_indices_to_exclude_from_models_filepath = meta_folder + 'weed_indices_to_exclude_from_models.json'

models_folder = '../models/'

predicted_segm_masks_folder = '../predictions/'

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

color_values = [
    [255, 255, 255],
    [0, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 0, 0]
]

hex_colors = ['#{:02X}{:02X}{:02X}'.format(r, g, b) for r, g, b in color_values]