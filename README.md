# Enhancing Keyphrase Extraction from Long Documents with Graph Embeddings
This is the code implementation of the paper "Enhancing Keyphrase Extraction from Long Documents with Graph Embeddings" by Martinez-Cruz, Mahata et al (https://arxiv.org/abs/2305.09316).

## Abstract
In this study, we investigate using graph neural network (GNN) representations to enhance contextualized representations of pre-trained language models (PLMs) for keyphrase extraction from lengthy documents. We show that augmenting a PLM with graph embeddings provides a more comprehensive semantic understanding of words in a document, particularly for long documents. We construct a co-occurrence graph of the text and embed it using a graph convolutional network (GCN) trained on the task of edge prediction. We propose a \textit{graph-enhanced} sequence tagging architecture that augments contextualized PLM embeddings with graph representations. Evaluating on benchmark datasets, we demonstrate that enhancing PLMs with graph embeddings outperforms state-of-the-art models on long documents, showing significant improvements in F1 scores across all the datasets. Our study highlights the potential of GNN representations as a complementary approach to improve PLM performance for keyphrase extraction from long documents.

## Keynotes
The submitted notebooks provide a comprehensive, step-by-step implementation of our method on the SemEval2010 dataset (available at https://huggingface.co/datasets/midas/semeval2010). Our approach leverages Bloomberg's KBIR language model (accessible at https://huggingface.co/bloomberg/KBIR) as the base model. Furthermore, we include a long document fine-tuning procedure for the model, excluding the use of graph embeddings, specifically for benchmarking purposes. The methodology we present can be easily replicated on different models and datasets by modifying the relevant parameters. The notebooks provided can be executed directly on Google Colab or on a local machine by installing the required dependencies listed in the requirements.txt file.

## Reference
@misc{martínezcruz2023enhancing,
      title={Enhancing Keyphrase Extraction from Long Scientific Documents using Graph Embeddings}, 
      author={Roberto Martínez-Cruz and Debanjan Mahata and Alvaro J. López-López and José Portela},
      year={2023},
      eprint={2305.09316},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
