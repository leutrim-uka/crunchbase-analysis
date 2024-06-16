# Crunchbase Analysis

## Overview
This repository contains analysis work on Crunchbase data. It includes scripts, notebooks, and configuration files necessary to perform topic modeling on Crunchbase datasets. This project relies on KeyBERT for keyword extraction, and on BERTopic and LLMs (Llama) for topic modeling. Classical NLP analysis in the notebooks relies on spaCy and NLTK.

## Structure
- **config**: Configuration files for the repository.
- **notebooks**: Jupyter notebooks with steps, explanations, and visualizations.
- **src**: Source code for the analysis.

### Notebooks
- `CB_analysis`: contains a general analysis of the company descriptions provided in the Crunchbase dataset. Visualizations provided with matplotlib.
- `CB_TopicModelling_LLM`: contains a straightforward implementation of BERTopic with Llama, which is also implemented using plain python files in the repo.

## Setup
To set up the environment for this analysis, run the `setup.sh` script provided in the repository. Due to the dataset size, a GPU is necessary. A RTX A5000 with 24GB of RAM from vast.ai appears to be sufficient for training with three batches.

`src/main.py` is the entrypoint.

Optional: Modify the `config.yaml` file to try different parameters for the models or change other settings.

**Note**: The Crunchbase data used for the analysis cannot be provided due to a non-disclosure agreement.

