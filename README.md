# Correcting Annotator Bias in Training Data: Population-Aligned Instance Replication (PAIR)

Aligning NLP Models with Target Population Perspectives using PAIR: Population-Aligned Instance Replication (https://aclanthology.org/2025.nlperspectives-1.9/)

## Content
- `data`: Data used for experiments is from [Kern et al. (2023)](https://aclanthology.org/2023.findings-emnlp.992.pdf).
- `results`: Folder containing evaluation results and plots after the training experiments.
- `results_filtered`: Folder containing evaluation results for difficult tweets (0.4 <=p_i<= 0.6) and plots after the training experiments.
- `test_results`: Folder containing evaluation results on gold test set after the training experiments.
- `eval_filtered.ipynb`: notebook script to evaluate models on difficult tweets (0.4 <=p_i<= 0.6).
- `eval_full_tweets.ipynb`: notebook script to evaluate models on full tweets.
- `simulation.ipynb`: notebook script to run the simulations.
- `train.py`: python script to training the models on the simulated data.

## Citation
```
@inproceedings{eckman-etal-2025-aligning,
    title = "Aligning {NLP} Models with Target Population Perspectives using {PAIR}: Population-Aligned Instance Replication",
    author = "Eckman, Stephanie  and
      Ma, Bolei  and
      Kern, Christoph  and
      Chew, Rob  and
      Plank, Barbara  and
      Kreuter, Frauke",
    editor = "Abercrombie, Gavin  and
      Basile, Valerio  and
      Frenda, Simona  and
      Tonelli, Sara  and
      Dudy, Shiran",
    booktitle = "Proceedings of the The 4th Workshop on Perspectivist Approaches to NLP",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.nlperspectives-1.9/",
    doi = "10.18653/v1/2025.nlperspectives-1.9",
    pages = "100--110",
    ISBN = "979-8-89176-350-0",
}
```