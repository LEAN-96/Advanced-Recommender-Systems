# Advanced Recommender Systems

## 1. Project Overview

This project involves the implementation of **Recommender Systems** using various machine learning techniques, aiming to solve the problem of **personalized recommendation**. The notebook demonstrates how to build a recommender system that predicts user preferences based on historical data. Users can expect to learn how to preprocess data, build models using collaborative filtering methods, and evaluate the performance of these models.

### Dataset

The dataset used in this project is the **MovieLens dataset**, which is a widely-used benchmark for movie recommendation systems.

- **Origin**: [MovieLens](https://grouplens.org/datasets/movielens/)
- **Size and Content**: The dataset contains:
  - **Users**: 943 users
  - **Movies**: 1,682 movies
  - **Ratings**: 100,000 ratings (on a scale from 1 to 5)
  - Each record consists of a `user_id`, `movie_id`, and `rating`.

### Processing Steps:
- **Handling Missing Data**: The dataset does not contain missing values, but if it did, we would handle them by either removing incomplete rows or imputing values.
- **Train Test Split**: We split the data into training and test sets for model evaluation, dividing them into two sets.

### Machine Learning Methods

The project implements two main algorithms for generating recommendations:

### Collaborative Filtering (KNN-based)

Collaborative filtering is a technique that predicts a user's preference based on the preferences of similar users or items. There are two main types:
1. **User-based Collaborative Filtering**: Recommends items that similar users liked.
2. **Item-based Collaborative Filtering**: Recommends items that are similar to items the user has liked.

In this project, we use the **K-Nearest Neighbors (KNN)** algorithm to find users or items that are closely related based on their rating patterns. For example, if User A has rated movies similarly to User B, we can recommend movies that User B liked to User A.

### Singular Value Decomposition (SVD)

SVD is a matrix factorization technique used to reduce the dimensionality of large, sparse matrices (like our user-item rating matrix). It decomposes the matrix into three smaller matrices.

By reducing the dimensions of these matrices, we can capture the most important relationships between users and items while ignoring noise or irrelevant data. SVD helps in making better predictions by generalizing well even with sparse data.

### Notebook Overview

The notebook is structured as follows:

### 1. **Data Loading and Preprocessing**
   - **Dataset Loading**: The MovieLens dataset is loaded, including user ratings and movie metadata.
   - **Train/Test Split**: The data is split into training and testing sets to evaluate model performance.
   - **Preprocessing**: Missing values are handled, and the data is transformed into a user-item matrix suitable for the recommendation algorithms.

2. **Model Building**
   - **Collaborative Filtering (KNN)**: A K-Nearest Neighbors model is built to find similar users or items based on their ratings.
   - **Matrix Factorization (SVD)**: Singular Value Decomposition (SVD) is applied to reduce the dimensionality of the user-item matrix, capturing latent features for better predictions.

3. **Model Training**
   - The models are trained using the training dataset. For KNN, the algorithm identifies the nearest neighbors, while SVD factorizes the matrix to learn user and item features.

4. **Evaluation**
   - The trained models are evaluated on the test dataset using metrics such as Mean Squared Error (MSE) to measure prediction accuracy.
   - The results are analyzed to determine how well the models perform in recommending movies to users.

5. **Generating Recommendations**
   - After training, recommendations are generated for specific users by predicting their ratings for unrated movies.
   - Example outputs are displayed, showing personalized movie suggestions for individual users.


## 2. Requirements

### Running Locally

To run the notebook locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LEAN-96/Advanced-Recommender-Systems.git
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
   Open the notebook (`2-Advanced_Recommender_Systems.ipynb`) in the Jupyter interface to run the notebook.

### Running Online via MyBinder

To run this notebook online without any local installation, you can use MyBinder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LEAN-96/Advanced-Recommender-Systems.git/HEAD?labpath=notebooks)

Once MyBinder loads:
1. Navigate to your notebook (`2-Advanced_Recommender_Systems.ipynb`) in the file browser on the left.
2. Click on the notebook to open it.
3. Run all cells using "Run All" (`Shift + Enter` for individual cells).

By using MyBinder, you can explore the notebook without installing any software locally.

## 3. Reproducing Results

To reproduce the results of this project:

1. Open the notebook (`2-Advanced_Recommender_Systems.ipynb`) using Jupyter or MyBinder.
2. Execute all cells sequentially by selecting them and pressing `Shift + Enter`.
3. Ensure that all cells execute without errors.
4. Observe output results including evaluation metrics and visualizations.

### Interpreting Results:

After training and evaluating the models, the notebook provides several metrics to assess the performance of different collaborative filtering techniques. The key metrics used are **Root Mean Squared Error (RMSE)** and **Mean Squared Error (MSE)**, which measure the difference between the predicted ratings and the actual ratings in the test set. A lower value for these metrics indicates better model performance.

### 1. **Memory-Based Collaborative Filtering**

Memory-based collaborative filtering relies on user or item similarity to make predictions. The RMSE values for both user-based and item-based collaborative filtering are as follows:

- **User-based CF RMSE: 3.124**
  - This value indicates how well the user-based collaborative filtering model predicts ratings based on similar users' preferences. An RMSE of 3.124 suggests that, on average, the predicted ratings deviate from the actual ratings by approximately 3.12 points.
  
- **Item-based CF RMSE: 3.453**
  - Item-based collaborative filtering uses similarities between items (movies) to make predictions. The RMSE of 3.453 shows that this model performs slightly worse than user-based CF, with a higher average error in its predictions.

### 2. **Model-Based Collaborative Filtering**

In model-based collaborative filtering, algorithms like matrix factorization are used to learn latent factors from the data. The evaluation metric provided here is:

- **Das Seltenheits-Niveau: 93.7%**
  - This metric reflects how well the model captures rare or unique patterns in user preferences. A higher percentage (93.7%) indicates that the model is effective at identifying less common user behaviors or preferences, which can be particularly useful for making personalized recommendations.

### 3. **SVD (Singular Value Decomposition)**

SVD is a matrix factorization technique used to reduce dimensionality and improve prediction accuracy by capturing latent features of users and items.

- **User-based CF MSE: 2.718**
  - The Mean Squared Error (MSE) for SVD is significantly lower than the RMSE values for memory-based methods, indicating that SVD provides more accurate predictions overall. An MSE of 2.718 suggests that the average squared difference between predicted and actual ratings is smaller, making SVD a more precise model in this case.

  ### Summary

- **User-based Collaborative Filtering** performs better than item-based CF in this case, as indicated by its lower RMSE.
- **SVD** outperforms both memory-based methods with a much lower MSE, suggesting it is better at capturing complex patterns in user-movie interactions.