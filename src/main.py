import logging
from utils import load_config, setup_logging, save_checkpoint
from topic_modeling import TopicModelingPipeline


def main():
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    setup_logging(config)
    logging.info("Starting the topic modeling pipeline...")

    pipeline = TopicModelingPipeline(config)
    docs, embeddings = pipeline.load_data()
    doc_chunks, embedding_chunks = pipeline.split_batches(docs, embeddings)
    topic_models = pipeline.process_batches(doc_chunks, embedding_chunks)
    merged_model = pipeline.merge_models(topic_models)

    save_checkpoint(merged_model, config['checkpoint']['dir'], "merged")
    logging.info("Topic modeling pipeline completed.")


if __name__ == "__main__":
    main()
