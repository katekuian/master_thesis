import numpy as np
import PIL
from PIL import Image
import datasets
import evaluate
import torch
import json
import codecs
import os
from os import sys

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer, EarlyStoppingCallback
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data_prepossessing import create_datasets_for_plants, get_labels
from constants import seed, weed_plants, models_folder
from config import model_type, crop


def init_image_processor(checkpoint):
    image_processor = SegformerImageProcessor.from_pretrained(checkpoint)
    return image_processor


def train_transforms(example_batch, image_processor):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def compute_metrics(num_labels, metric, eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics
    

def init_training_arguments(prediction_loss_only):
    return TrainingArguments(
        output_dir="segformer-b0-scene-parse-150",
        learning_rate=6e-5,
        num_train_epochs=100,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=30,
        eval_steps=30,
        logging_steps=1,
        prediction_loss_only=prediction_loss_only,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        seed=seed,
    )


def init_training_arguments_for_training():
    return init_training_arguments(True)


def init_training_arguments_for_evaluation():
    return init_training_arguments(False)


def initialize_trainer(model, training_args, num_labels, metric, train_ds, test_ds):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=lambda eval_pred: compute_metrics(num_labels, metric, eval_pred),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )


def train_model_of_type_for_crop(model_type, crop):
    # Define a model checkpoint to be finetuned
    checkpoint = "nvidia/mit-b0"

    # Prepare the data for the model training
    model_plant_names = [crop] + weed_plants
    train_ds, test_ds = create_datasets_for_plants(model_plant_names, model_type, crop)

    print("Training subset number of images: " + str(train_ds.num_rows))
    print("Test subset number of images: " + str(test_ds.num_rows))

    image_processor = init_image_processor(checkpoint)
    train_ds.set_transform(lambda example_batch: train_transforms(example_batch, image_processor))
    test_ds.set_transform(lambda example_batch: train_transforms(example_batch, image_processor))

    # Generate labels for the model
    id2label, label2id = get_labels(crop, model_type)
    num_classses = len(id2label)

    print('Number of classes:', num_classses)
    print('id2label:', id2label)
    print('label2id:', label2id)

    # Initialize and train model
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
    training_args_for_training = init_training_arguments_for_training()
    metric = evaluate.load("mean_iou")
    trainer = initialize_trainer(model, training_args_for_training, num_classses, metric, train_ds, test_ds)
    trainer.train()

    # Save the trained model, so that it can be used for inference later
    trainer.save_model(models_folder + model_type + '/' + crop)
    
    # Save the log history, so that it can be used for plotting later
    with open(models_folder + model_type + '/' + crop + '/log_history.json', 'w') as file:
        log_history = trainer.state.log_history
        json.dump(log_history, file)

    # Instantiate new trainer for evaluation that will use compute_metrics method
    training_args_for_evaluation = init_training_arguments_for_evaluation()
    eval_trainer = initialize_trainer(trainer.model, training_args_for_evaluation, num_classses, metric, train_ds, test_ds)
    test_metric = eval_trainer.evaluate(test_ds)
    with open(models_folder + model_type + '/' + crop + '/test_metric.json', 'w') as file:
        json.dump(test_metric, file)



def train_model_from_config():
    train_model_of_type_for_crop(model_type, crop)


if __name__ == '__main__':
    print(train_model_from_config())