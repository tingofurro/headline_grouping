# HeadLine Grouping

This repository will contain the dataset and models described in NAACL2021 paper: [News Headline Grouping as a Challenging NLU Task](https://people.eecs.berkeley.edu/~phillab/pdfs/NAACL2021_HLG.pdf).

<p align="center">
  <img width="445" height="233" src="https://people.eecs.berkeley.edu/~phillab/images/naacl2021_headline_grouping_example.png"><br />
  Example Headline Groups from the <i>International Space Station</i> timeline.
</p>

## Dataset Releases

[In the release](https://github.com/tingofurro/headline_grouping/releases/tag/0.1), we provide two versions of the HLGD:
- The original annotation of the 10 timelines present in HLGD, with annotator identities anonymized. For each headline, we populate: the URL of the headline (`url`), the headline text (`headline`), the publication date (`date`), the group annotations of 5 annotators (`annot_1_group`, ... `annot_5_group`), and an aggregate group (`global_group`).
- The classification compatible version of HLGD containing ~20,000 headline pairs and binary labels indicating whether the headlines are in the same global group or not. The classification dataset is integrated into HuggingFace's `datasets` library. The dataset can be loaded in the following way:
```
!pip install datasets
from datasets import load_dataset
data = load_dataset('hlgd')
```

Note: We considered the legal component of the release of HLGD, and consider that the release of the dataset falls under fair use. [See more detail here](https://github.com/tingofurro/headline_grouping/blob/main/LEGAL.md)

## Model Releases

We release two models:

- `classifier_electra_hlpc_1_f1_0.7442.bin` model corresponds to the `Electra Finetune on HLGD + Time` in the paper. It is compatible with a HuggingFace `AutoModelForSequenceClassification` model, using the model card: `google/electra-small-discriminator`

- `gpt2med_headline_gen_1.645.bin` model corresponds to the headline generator used for the `Headline Generator Swap` results. It can be used in conjunction with the `model_generator_swap.py` file.

## Cite the work

If you make use of the code, models, or algorithm, please cite our paper:
```
@inproceedings{Laban2021NewsHG,
  title={News Headline Grouping as a Challenging NLU Task},
  author={Laban, Philippe and Bandarkar, Lucas and Hearst, Marti A},
  booktitle={NAACL 2021},
  publisher = {Association for Computational Linguistics},
  year={2021}
}
```

## Contributing

If you'd like to contribute, or have questions or suggestions, you can contact us at phillab@berkeley.edu.
All contributions welcome!
