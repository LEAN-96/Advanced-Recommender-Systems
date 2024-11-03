# Recommender Systems

## 1. Project Overview

This project involves the implementation of **Recommender Systems** using various machine learning techniques, aiming to solve the problem of **personalized recommendation**. The notebook demonstrates how to build a recommender system that predicts user preferences based on historical data. Users can expect to learn how to preprocess data, build models using collaborative filtering methods, and evaluate the performance of these models.

### Dataset

The dataset used in this project is a **movie rating dataset** that contains user ratings for different movies. It includes information such as:
- **Users**: Unique identifiers for users.
- **Movies**: Unique identifiers for movies.
- **Ratings**: Numerical ratings provided by users for specific movies.

The dataset undergoes preprocessing steps, including:
- Handling missing values.
- Normalizing or scaling ratings.
- Splitting the data into training and test sets for model evaluation.

### Machine Learning Methods

The following machine learning methods are applied in the notebook:

- **Collaborative Filtering (Matrix Factorization)**: This method predicts user preferences by learning latent factors from user-item interactions. It is widely used in recommendation systems to predict ratings for unseen items.
  
- **K-Nearest Neighbors (KNN)**: KNN is used here for item-based collaborative filtering, where the similarity between items is calculated to recommend items similar to those a user has rated highly.

### Notebook Overview

The notebook is structured as follows:

1. **Data Loading and Preprocessing**:
   - The dataset is loaded from a CSV file.
   - Preprocessing steps include handling missing data, normalizing ratings, and splitting the data into training and test sets.

2. **Model Building**:
   - The notebook defines two main models:
     - A matrix factorization model using Singular Value Decomposition (SVD).
     - A KNN-based collaborative filtering model.
   
3. **Model Training**:
   - The SVD model is trained using stochastic gradient descent to minimize the error between predicted and actual ratings.
   - The KNN model is trained by calculating item-item similarities based on user ratings.

4. **Evaluation**:
   - The models are evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) to assess prediction accuracy.
   
5. **Visualization**:
   - Users can expect visualizations such as:
     - Accuracy plots comparing training and testing errors.
     - Confusion matrices showing correct vs incorrect predictions for classification-based recommendations.

## 2. Requirements

### Running Locally

To run the notebook locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LEAN-96/Recommender-Systems.git
    cd recommender-system
    ```

2. **Set up a virtual environment**:
    Using `venv`:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

    Or using `conda`:
    ```bash
    conda create --name ml-env python=3.8
    conda activate ml-env
    ```

3. **Install project dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
   Open the notebook (`1-Recommender_Systems.ipynb`) in the Jupyter interface to run the notebook.

### Running Online via MyBinder

To run this notebook online without any local installation, you can use MyBinder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LEAN-96/Recommender-Systems.git/HEAD?labpath=notebooks)

Once MyBinder loads:
1. Navigate to your notebook (`1-Recommender_Systems.ipynb`) in the file browser on the left.
2. Click on the notebook to open it.
3. Run all cells using "Run All" (`Shift + Enter` for individual cells).

By using MyBinder, you can explore the notebook without installing any software locally.

## 3. Reproducing Results

To reproduce the results of this project:

1. Open the notebook (`1-Recommender_Systems.ipynb`) using Jupyter or MyBinder.
2. Execute all cells sequentially by selecting them and pressing `Shift + Enter`.
3. Ensure that all cells execute without errors.
4. Observe output results including evaluation metrics and visualizations.

### Interpreting Results:

- **Accuracy Metrics**: These metrics show how well each model performs in predicting user ratings, with lower RMSE indicating better performance.
  
- **Confusion Matrix**: For classification-based recommendations, this visualization helps in understanding how many correct vs incorrect predictions were made.

- **Feature Analysis or Graphs**: Visual representations of latent factors or item similarities may be included to help interpret model behavior.