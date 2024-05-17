# Proste Cancer Aggressiveness Index (PCAI)
This repository includes the main building blocks of PCAI. Since it is based on tissue microarray (TMA) spots of real patients, the original dataset is not included and replaced by a [toy dataset](data/tma_dataset/). 

## How to run
The repository requires python 3.10, additional python libraries, and some system-wide packages. They can be installed (e.g. inside a docker container) using the following commands

```bash
# clone project
git clone 'this repository' pcai
cd pcai

# create conda environment
conda create -n env-pcai python=3.10 -y
conda activate env-pcai

# install libgcc for openslide
conda install -c anaconda glib
conda install -c conda-forge gcc=12.1.0

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install apt requirements
xargs sudo apt-get install -y < requirements.apt

# install python requirements
pip install -r requirements.txt
```

# Sample notebooks
The following notebooks are included to provide a basic understanding of how the PCAI model works.


## Patch Loading

As demonstrated in [patch_loader.ipynb](notebook/patch_loader.ipynb), the patch loaded based on [Openslide](https://openslide.org/api/python/) allows patch-wise extraction of individual patches from whole slide images. An example is shown below where four individual patches are illustrated along with an overall toy "slide":

![image](notebook/output/patch_loader_example.png)

## Cancer Indicator model
The Cancer Indicator (CI) model can be used to preselect patches of relevant parts of biopsy slides. It is trained on the PANDA dataset to predict patch-wise labels of cancerous vs healthy tissue. The following notebooks show how the model is trained an evaluated. Note that only a few slides of the whole PANDA dataset are included in this repository. For the full dataset, see [https://panda.grand-challenge.org/](https://panda.grand-challenge.org/).

### Label Extraction

The [01_patch_extraction.ipynb](notebook/cancer_indicator/01_patch_extraction.ipynb) notebook shows how individual patches with corresponding labels (0=healthy, 1=cancerous) tissue are derived. For whole slides, the extracted patches with corresponding coordinates and a label overlay are shown:

<table>
    <tr>
      <td>
        <figure class="image">
            <img src="notebook/cancer_indicator/output/patch_mask_overview_1.png">
            <figcaption>A biopsy that does not show cancerous tissue. Only patches that show healthy tissue are extracted (green).</figcaption>
        </figure>
      </td>
      <td>
        <figure class="image">
            <img src="notebook/cancer_indicator/output/patch_mask_overview_3.png">
            <figcaption>A biopsy that contains a cancerous and non-cancerous area. Patches with healthy tissue (green) and patches with cancerous tissue (red) can be extracted.</figcaption>
        </figure>
      </td>
    </tr>
  </table>

### Model Training

In [02_training.ipynb](notebook/cancer_indicator/02_training.ipynb), a cancer indicator model is trained and evaluated on a few exemplary slides. The resulting model provides patch-level predictions for individual patches as shown below. Note that the results are derived from a model that only uses 24 input images for training.

<figure class="image">
    <img src="notebook/cancer_indicator/output/ci_prediction_examples.png">
    <figcaption>Exemplary patches with low (left) and high (right) predictions of the CI model. The originally extracted label of the patches is indicated by the red or green outline of each patch.</figcaption>
</figure>

## PCAI
This part presents some notebooks on how to utilize the PCAI code

### Masking
TODO

### Prepare a slide
TODO

### Build a Dataset
TODO

### Build a Data Loader
TODO

### Train a Model
TODO

