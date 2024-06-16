import logging
import sys

from utils import load_config, setup_logging, save_checkpoint
from topic_modeling import TopicModelingPipeline


def main():
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    setup_logging(config)
    logging.info("Starting the topic modeling pipeline...")

    pipeline = TopicModelingPipeline(config)
    docs = pipeline.load_data()
    doc_chunks = pipeline.split_batches(docs)
    topic_models = pipeline.process_batches(doc_chunks)
    merged_model = pipeline.merge_models(topic_models)

    if merged_model is None:
        logging.error("No models to merge. Exiting...")
        sys.exit(0)

    save_checkpoint(merged_model, config['checkpoint']['dir'], "merged")
    logging.info("Topic modeling pipeline completed.")


if __name__ == "__main__":
    main()
