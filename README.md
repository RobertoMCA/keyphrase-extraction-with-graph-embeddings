# Enhancing Keyphrase Extraction from Long Documents with Graph Embeddings
This code implementation corresponds to the paper titled "Enhancing Keyphrase Extraction from Long Documents with Graph Embeddings" authored by Martinez-Cruz, Mahata et al. [Link to paper](https://arxiv.org/abs/2305.09316).

## Abstract
In this study, we investigate using graph neural network (GNN) representations to enhance contextualized representations of pre-trained language models (PLMs) for keyphrase extraction from lengthy documents. We show that augmenting a PLM with graph embeddings provides a more comprehensive semantic understanding of words in a document, particularly for long documents. We construct a co-occurrence graph of the text and embed it using a graph convolutional network (GCN) trained on the task of edge prediction. We propose a \textit{graph-enhanced} sequence tagging architecture that augments contextualized PLM embeddings with graph representations. Evaluating on benchmark datasets, we demonstrate that enhancing PLMs with graph embeddings outperforms state-of-the-art models on long documents, showing significant improvements in F1 scores across all the datasets. Our study highlights the potential of GNN representations as a complementary approach to improve PLM performance for keyphrase extraction from long documents.

## Keynotes
The submitted notebooks provide a detailed, step-by-step implementation of our keyphrase extraction method for long documents using the [SemEval2010](https://huggingface.co/datasets/midas/semeval2010) dataset and Bloomberg's language model [KBIR](https://huggingface.co/bloomberg/KBIR) as the basis for the examples. Additionally, we provide a long document fine-tuning procedure for the model, excluding the use of graph embeddings, specifically designed for benchmarking purposes. The methodology we present allows for easy replication on various models and datasets by adjusting the relevant parameters. These notebooks are compatible with both Google Colab and local machines, meeting or exceeding the necessary specifications, ensuring seamless execution on either platform without any compatibility concerns.

## Link to Google Colab notebooks
- [Enhancing Keyphrase Extraction from Long Documents with Graph Embeddings - Fintune KBIR with Graph Embeddings in SemEval2010](https://colab.research.google.com/drive/17iseCNZoKJCsoQciMESwTL34sGjmwmqc)
- [Enhancing Keyphrase Extraction from Long Documents with Graph Embeddings - Fintune KBIR in SemEval2010](https://colab.research.google.com/drive/1zaUWggRtEauzhbKmNUALrdoC3k7FRqEx)

## Reference
@misc{martínezcruz2023enhancing,
      title={Enhancing Keyphrase Extraction from Long Scientific Documents using Graph Embeddings}, 
      author={Roberto Martínez-Cruz and Debanjan Mahata and Alvaro J. López-López and José Portela},
      year={2023},
      eprint={2305.09316},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
