from PIL import Image
import datasets
import json
import codecs
import constants


def get_image_meta_filepath(plant_name):
    suffix = '_images.json'
    if plant_name in constants.weed_plants:
        suffix = '_no_crop_images.json'
    return constants.meta_folder + plant_name + suffix


def get_images_to_exclude():
    return json.load(codecs.open(constants.images_to_exclude_from_models_filepath, 'r', 'utf-8-sig'))    


def get_image_list_for_plant(plant_name, model_type, crop):
    # Create an empty list to store the dataset
    image_list = []
    plant_image_names = json.load(codecs.open(get_image_meta_filepath(plant_name), 'r', 'utf-8-sig'))

    # Exclude images that contain more than one crop
    image_names_to_exclude = get_images_to_exclude()
    plant_image_names = [image_name for image_name in plant_image_names if image_name not in image_names_to_exclude]

    # Iterate over the image and annotation paths
    for image_name in plant_image_names:
        # Load the image using PIL with a 'with' statement
        with Image.open(constants.image_folder + image_name) as image:
            path = None
            if model_type == 'multiclass':
                path = constants.annotations_multiclass_folder + crop + '/' + image_name
            elif model_type == 'binary':
                path = constants.annotations_binary_folder + crop + '/' + image_name

            # Load the annotation using PIL with a 'with' statement
            with Image.open(path) as annotation:
                # Create a dictionary entry for the dataset
                entry = {'image': image, 'annotation': annotation}

                # Add the entry to the dataset
                image_list.append(entry)

    return image_list


def create_and_split_dataset_for_plant(plant_image_list):
    dataset = datasets.Dataset.from_list(plant_image_list)
    train_ds, test_ds = dataset.train_test_split(test_size=0.5, seed=constants.seed).values()
    
    return train_ds, test_ds


def create_datasets_for_plants(plant_names, model_type, crop):
    p0_image_list = get_image_list_for_plant(plant_names[0], model_type, crop)
    train_ds, test_ds = create_and_split_dataset_for_plant(p0_image_list)

    for plant_name in plant_names[1:]:
        p_image_list = get_image_list_for_plant(plant_name, model_type, crop)
        p_train_ds, p_test_ds = create_and_split_dataset_for_plant(p_image_list)

        train_ds = datasets.concatenate_datasets([train_ds, p_train_ds])
        test_ds = datasets.concatenate_datasets([test_ds, p_test_ds])

    return train_ds, test_ds


def get_labels(crop, model_type):
    labels = ['void', 'soil', crop]

    if model_type == 'binary':
        labels.append('weeds')
    elif model_type == 'multiclass':
        weed_indices_to_exclude = json.load(codecs.open(constants.weed_indices_to_exclude_from_models_filepath, 'r', 'utf-8-sig'))
        weed_indices_to_consider = [x for x in constants.weed_indices if x not in weed_indices_to_exclude]
        for weed_index in weed_indices_to_consider:
            weed_name = list(constants.plant_classification.keys())[weed_index]
            labels.append(weed_name.replace(" ", "_"))

    ids = list(range(0, len(labels)))

    id2label = dict(zip(ids, labels))
    label2id = dict(zip(labels, ids))

    return id2label, label2id