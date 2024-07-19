# Authorship Attribution on Short Texts in Slovenian Language

### [Link to the paper](https://doi.org/10.3390/app131910965)

This repository contains the code used in the project titled "Authorship Attribution on Short Texts in Slovenian Language".
The project investigates the task of authorship attribution on short texts in Slovenian using two BERT language models.

The code can be used to fine-tune BERT models on short texts in Slovenian for authorship attribution.
The datasets used in this project are publicly available on Huggingface Datasets and the trained
Slovenian BERT models are available on Huggingface Models.

The project demonstrates the feasibility of using BERT models for authorship attribution in short texts and
provides a solid starting point for systems tackling the growing problem of misinformation, hate speech,
and the manipulation of public opinion.

## Installation
To install the necessary packages, you have two options: either use the `requirements.txt` file or use Poetry.

If you use Poetry, you can install the necessary packages by running the following command:

```bash
poetry install
```

If you use the `requirements.txt` file, you can install the necessary packages by running the following command:

```bash
pip install -r requirements.txt
```

## Usage
### Generating the IMDb1m datasets
To generate the IMDb1m dataset to use with our trainer code, perform the following steps:

1. Download the IMDb1m dataset from [here](https://umlt.infotech.monash.edu/?page_id=266) (make sure to download from the link with the title `IMDb 1 million`), extract the files and set the
   `review_texts_file_path` and `posts_texts_file_path` config variables in the
   [config.yaml file](aa_slovenian/dataset/imdb1m/config.yaml)
2. Enter your HuggingFace credentials into the config file.
3. Run the [main.py file](aa_slovenian/dataset/imdb1m/main.py) to generate the dataset.

### Generating the RTV SLO datasets
1. Run [main.py](aa_slovenian/dataset/rtv_slo/main.py) to retrieve the comment data from the RTV SLO website.
2. Run [prepare_dataset.py](aa_slovenian/dataset/rtv_slo/prepare_dataset.py) to prepare the dataset for training.
3. Upload the dataset to HuggingFace Datasets manually.

### Training the models
To train the models, run [trainer.py](aa_slovenian/train/trainer.py).
When running the file, you have to specify the following arguments:
- `dataset_size`: The number of the authors in the dataset (`5, 10, 20, 50 or 100` for RTV SLO, and `5, 10, 25, 50 or 100` for IMDb1m)
- `has_ooc`: Whether the dataset has an out-of-corpus author (`0` for false, `1` for true)
- `use_multilingual_bert`: Whether to use the multilingual BERT model (`0` for false/SloBERTa, `1` for true/mBERT)
- `use_imdb`: Whether to use the IMDb1m dataset (`0` for false/RTV SLO, `1` for true/IMDb1m)

Before starting the script, you have to edit the `huggingface_username` variable in line 11.

## Citation

You can cite the paper using the following BibTeX snippet:

```bibtex
@article{gabrovvsek2023authorship,
  title={Authorship Attribution on Short Texts in the Slovenian Language},
  author={Gabrov{\v{s}}ek, Gregor and Peer, Peter and Emer{\v{s}}i{\v{c}}, {\v{Z}}iga and Batagelj, Borut},
  journal={Applied Sciences},
  volume={13},
  number={19},
  pages={10965},
  year={2023},
  publisher={MDPI}
}
```
