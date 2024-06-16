import os
import yaml
import logging
from pathlib import Path
from bertopic import BERTopic


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(config):
    logging.basicConfig(level=config['logging']['level'], format=config['logging']['format'])


def ensure_directory_exists(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, checkpoint_dir, batch_index):
    checkpoint_path = os.path.join(checkpoint_dir, f"topic_model_{batch_index}.pkl")
    model.save(checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_dir, batch_index):
    checkpoint_path = os.path.join(checkpoint_dir, f"topic_model_batch_{batch_index}.pkl")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        return BERTopic.load(checkpoint_path)
    return None
