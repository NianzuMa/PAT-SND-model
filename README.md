# Semantic Novelty Detection and Characterization in Factual Text Involving Named Entities

* This repository is the original implementation of EMNLP 2022 Paper: [Semantic Novelty Detection and Characterization in Factual Text Involving Named Entities](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.627/)
* Please contact [@NianzuMa](https://github.com/NianzuMa) for questions and suggestions.
* For code running issues, please submit to the Github issues of this repository.
* The dataset can be downloaded this [drive](https://drive.google.com/drive/folders/123Ohexcrz3jq3mIYADdpaGKpGyqlg05h?usp=share_link).


## Citation

Nianzu Ma, Sahisnu Mazumder, Alexander Politowicz, Bing Liu, Eric Robertson, and Scott Grigsby. 2022. Semantic Novelty Detection and Characterization in Factual Text Involving Named Entities. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9225–9252, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

```
@inproceedings{ma2022semantic,
  title="Semantic Novelty Detection and Characterization in Factual Text Involving Named Entities",
  author="Ma, Nianzu and Mazumder, Sahisnu and Politowicz, Alexander and Liu, Bing and Robertson, Eric and Grigsby, Scott",
  booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
  month = dec,
  year={2022},
  address = "Online and Abu Dhabi, United Arab Emirates",
  publisher = "Association for Computational Linguistics",
  url = "https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.627/",
    pages = "9225–-9252",
    abstract = "Much of the existing work on text novelty detection has been studied at the topic level, i.e., identifying whether the topic of a document or a sentence is novel or not. Little work has been done at the fine-grained semantic level (or contextual level). For example, given that we know Elon Musk is the CEO of a technology company, the sentence “Elon Musk acted in the sitcom The Big Bang Theory” is novel and surprising because normally a CEO would not be an actor. Existing topic-based novelty detection methods work poorly on this problem because they do not perform semantic reasoning involving relations between named entities in the text and their background knowledge. This paper proposes an effective model (called PAT-SND) to solve the problem, which can also characterize the novelty. An annotated dataset is also created. Evaluation shows that PAT-SND outperforms 10 baselines by large margins.",
}
```