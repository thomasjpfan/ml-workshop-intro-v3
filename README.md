# Introduction to scikit-learn: Machine Learning in Python

*By Thomas J. Fan*

[Link to slides](https://thomasjpfan.github.io/ml-workshop-intro-v3/)

Scikit-learn is a Python machine learning library used by data science practitioners from many disciplines. We start this training by learning about scikit-learn's API for supervised machine learning. scikit-learn's API mainly consists of three methods: fit to build models, predict to make predictions from models, and transform to modify data. This consistent and straightforward interface helps to abstract away the algorithm, thus allowing us to focus on our particular problems. We explore the preprocessing techniques on numerical, categorical data, and missing data. We see how to use the Pandas output API to see how to use Panda DataFrames with scikit-learn. Lastly, we cover the Pipeline, which enables us to connect transformers with a classifier or regressor to build a data flow where the output of one layer is the input of another. After this training, you will have the foundations to apply scikit-learn to your machine-learning problems.

## Obtaining the Material

### Run with Google's Colab

You can run the notebooks on Google's Colab:

1. [Supervised learning with scikit-learn](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intro-v3/blob/main/notebooks/01-supervised-learning.ipynb)
2. [Preprocessing](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intro-v3/blob/main/notebooks/02-preprocessing.ipynb)
3. [Pipelines](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intro-v3/blob/main/notebooks/03-pipelines.ipynb)
4. [Categorical Data](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intro-v3/blob/main/notebooks/04-categorical-data.ipynb)

### With git

The most convenient way to download the material is with git:

```bash
git clone https://github.com/thomasjpfan/ml-workshop-intro-v3
```

Please note that I may add and improve the material until shortly before the session. You can update your copy by running:

```bash
cd ml-workshop-intro-v3
git pull origin main
```

### Download zip

If you are not familiar with git, you can download this repository as a zip file at: [github.com/thomasjpfan/ml-workshop-intro-v3/archive/main.zip](https://github.com/thomasjpfan/ml-workshop-intro-v3/archive/main.zip). Please note that I may add and improve the material until shortly before the session. To update your copy please re-download the material a day before the session.

## Running the notebooks

### Local Installation

Local installation requires `conda` to be installed on your machine. The simplest way to install `conda` is to install `miniconda` by using an installer for your operating system provided at [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). After `conda` is installed, navigate to this repository on your local machine:

```bash
cd ml-workshop-intro-v3
```

Then download and install the dependencies:

```bash
conda env create -f environment.yml
```

This will create a virtual environment named `ml-workshop-intro-v3`. To activate this environment:

```bash
conda activate ml-workshop-v3
```

Finally, to start `jupyterlab` run:

```bash
jupyter lab
```

This should open a browser window with the `jupterlab` interface.

## License

This repo is under the [MIT License](LICENSE).
