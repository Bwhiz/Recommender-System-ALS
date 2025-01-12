# Personalized Recommendation System with Matrix Factorization and Bias Terms

This repository contains the code and resources for building a personalized movie recommendation system using matrix factorization, incorporating both latent factors and bias terms. The model is applied to the MovieLens 25M dataset and evaluated on different hyperparameters and configurations.

## Project Structure
```
├── assets/                   # Generated images (e.g.visualizations )
├── main.ipynb                # Main notebook for project work on the 25M dataset
├── ml-25M/                   # MovieLens 25M dataset (excluded from the repo due to size)
├── ml-latest-small/          # Small dataset used for prototyping (excluded from the repo due to size)
├── models/                   # Model files (pickled models, excluded from the repo due to size)
├── prototype.ipynb           # Prototype code used for quick testing on small dataset
├── README.md                 # This file
├── requirements.txt          # Python dependencies
```


## Overview

This project explores matrix factorization techniques to build a movie recommendation system. The approach includes:
- **Matrix Factorization**: Used to capture latent factors for users and items.
- **Bias Terms**: Incorporated for both users and items to improve model accuracy.
- **Evaluation**: The model is tested using RMSE and the results are compared across different embedding dimensions (`k`).

The full MovieLens 25M dataset was used for the final model, while a smaller dataset (`ml-latest-small`) was used for prototyping.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Bwhiz/Recommender-System-ALS.git
   cd Recommender-System-ALS
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
3. Download the datasets:

    The datasets (ml-25M and ml-latest-small) are too large to be included in the repository. You can download the 25M dataset from [here](https://grouplens.org/datasets/movielens/25m/), while you can download the small dataset [here](https://grouplens.org/datasets/movielens/latest/).
    Place them in the respective folders.
4. The pickle files would be generated when the `main.ipynb` file is executed and it would be saved in the 'models/' folder.

## Running the Notebooks

- Prototype.ipynb: This notebook contains the code for prototyping the recommendation system on the smaller dataset (`ml-latest-small`).

- Main.ipynb: This notebook contains the complete workflow, including data preprocessing, model building, and evaluation on the full MovieLens 25M dataset.

## Results

The results from the final model, including the effects of different embedding dimensions (`k`), can be found in the generated images in the `assets/` folder.