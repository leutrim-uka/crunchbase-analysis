import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, LlamaCPP
from llama_cpp import Llama
from utils import save_checkpoint, load_checkpoint, ensure_directory_exists


class TopicModelingPipeline:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config['checkpoint']['dir']
        ensure_directory_exists(self.checkpoint_dir)
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en")
        self.umap_model = UMAP(**config['model']['umap'])
        self.hdbscan_model = HDBSCAN(**config['model']['hdbscan'])
        self.llm = self._load_llm()
        self.representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "LLM": LlamaCPP(self.llm, prompt=self._get_prompt()),
        }
        self.batch_size = config['model']['batch_size']

    def _load_llm(self):
        return Llama(
            model_path=self.config['llm']['model_path'],
            n_gpu_layers=self.config['llm']['n_gpu_layers'],
            n_ctx=self.config['llm']['n_ctx'],
            stop=self.config['llm']['stop']
        )

    def _get_prompt(self):
        return """ Q:
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the above information, can you generate one to three classes of typical events in the given setting?
A:
"""

    def load_data(self):
        data_dir = self.config['data']['dir']
        embeddings_path = os.path.join(data_dir, self.config['data']['embeddings'])
        descriptions_csv_path = os.path.join(data_dir, self.config['data']['descriptions'])
        embeddings = np.load(embeddings_path)
        df = pd.read_csv(descriptions_csv_path)
        docs = df['description'].dropna().astype(str).tolist()
        assert len(docs) == len(embeddings), "Number of documents must match the number of embeddings."
        return docs, embeddings

    def split_batches(self, docs, embeddings):
        doc_chunks = [docs[i:i + self.batch_size] for i in range(0, len(docs), self.batch_size)]
        embedding_chunks = [embeddings[i:i + self.batch_size] for i in range(0, len(embeddings), self.batch_size)]
        return doc_chunks, embedding_chunks

    def process_batches(self, doc_chunks, embedding_chunks):
        topic_models = []

        # process in batches and merge models, otherwise runs out of GPU RAM
        for batch_index, (batch_docs, batch_embeddings) in enumerate(
                tqdm(zip(doc_chunks, embedding_chunks), total=len(doc_chunks), desc="Processing batches")):
            topic_model = load_checkpoint(self.checkpoint_dir, batch_index)
            if topic_model is None:
                topic_model = self._create_topic_model()
                topic_model.fit(batch_docs, embeddings=np.array(batch_embeddings))
                save_checkpoint(topic_model, self.checkpoint_dir, batch_index)
            topic_models.append(topic_model)

        return topic_models

    def _create_topic_model(self):
        return BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            representation_model=self.representation_model,
            top_n_words=10,
            verbose=True
        )

    def merge_models(self, topic_models):
        merged_model = BERTopic.merge_models(topic_models)
        merged_model.save("merged_topic_model")
        return merged_model

    def transform_documents(self, model, docs, embeddings):
        return model.transform(docs, embeddings=embeddings)
