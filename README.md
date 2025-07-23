# Professor Rating Prediction 🎓

This project focuses on predicting professor ratings at the University of Maryland (UMD) using machine learning models. The goal is to leverage publicly available data from PlanetTerp, a website where UMD students rate professors, to predict a professor's average rating based on various features, excluding their actual rating. This project was part of my CMSC320: Introduction to Data Science course at UMD, and it could potentially be extended to university departments for evaluating new hires.

## Project Overview

The project involves a comprehensive machine learning pipeline:
1.  **Data Collection:** Extracting professor and course data from the PlanetTerp API.
2.  **Feature Engineering:** Creating relevant features, including sentiment analysis of student reviews.
3.  **Model Training:** Implementing and training various regression models.
4.  **Evaluation:** Assessing model performance using key metrics and cross-validation.

## Files Included

* **`professor_rating_predictions.ipynb`**: A Jupyter Notebook (Google Colab format) containing all the Python code for data acquisition, preprocessing, feature engineering, model training, and evaluation. This notebook provides a detailed, executable walkthrough of the entire project.
* **`predicting_ratings_presentation.pdf`**: A PDF presentation summarizing the project's objectives, methodology, data sources, models used, evaluation metrics, and key conclusions. This is ideal for a quick overview of the project.

## Project Goal

The primary problem addressed is to predict professor ratings using information *other than* their actual star ratings. This means relying on features derived from course GPAs, the number of courses taught, and the sentiment of student reviews.

## Data Collection & Preprocessing

Data was collected from the [PlanetTerp API](https://planetterp.com/api/) using Python's `requests` library and JSON parsing. Due to API limits, a looping mechanism was implemented to fetch data in batches of 100 items.

**Key Data Points Collected:**
* **Professors:** Name, Slug, Type, Courses taught, Average Rating (target variable), and Reviews.
* **Courses:** Average GPA, Professors associated, Department, Course Number, Name, Title, Recent status, and Gen-Eds.

**Feature Engineering:**
* **Sentiment Analysis:** Utilized the `transformers` library's sentiment-analysis pipeline to classify student reviews as "positive" or "negative". This provided `neg_reviews` and `pos_reviews` counts for each professor.
* **Average Expected Grade:** Calculated the average expected letter grade (converted to a 4.0 GPA scale) from student reviews.
* **Average Course GPA:** Computed the average GPA of all courses a professor has taught.
* **Number of Courses:** Counted the total number of courses taught by each professor.
* **Number of Ratings:** Sum of positive and negative reviews.

A total of **4177 professors** were included in the final dataset after filtering out those without reviews.

## Models Implemented

Several regression models from `scikit-learn` were implemented and evaluated:

* **Random Forest Regressor:** An ensemble method using multiple decision trees, averaging their predictions to improve accuracy and reduce overfitting.
* **K-Nearest Neighbors (KNN) Regressor:** A non-parametric, instance-based learning algorithm that predicts a value based on the average of its `k` nearest neighbors. The optimal `k` value (8) was determined using an "elbow" method on an RMSE vs. K plot.
* **Support Vector Regressor (SVR):** An extension of Support Vector Machines for regression tasks, aiming to find the best hyperplane that fits the data with a maximal margin.
* **Decision Tree Regressor:** A tree-structured model that makes predictions by splitting data based on features. While simple and interpretable, they are prone to overfitting.
* **Elastic Net Regression:** A linear regression model that combines L1 (Lasso) and L2 (Ridge) regularization penalties, useful for feature selection and handling multicollinearity.

## Evaluation & Results

Model performance was evaluated using:
* **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors.
* **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values.
* **R-squared ($R^2$) Score:** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

**Ten-Fold Cross-Validation** was used for a robust evaluation, averaging the metrics across 10 different splits of the data.

| Model           | Average RMSE | Average MAE | Average $R^2$ |
| :-------------- | :----------- | :---------- | :------------ |
| **Random Forest** | **0.80687** | **0.56328** | **0.49483** |
| KNN             | 0.82704      | 0.59395     | 0.46787       |
| SVR             | 0.85856      | 0.58717     | 0.42879       |
| Decision Trees  | 1.05981      | 0.70775     | 0.12605       |
| Elastic Net     | 1.07299      | 0.86986     | 0.10942       |

**Conclusion:**
* **Random Forest Regressor** emerged as the best-performing model, achieving the lowest RMSE and MAE, and the highest $R^2$ score.
* The sentiment analysis features (positive and negative review counts) were found to be the most important predictors of professor ratings.
* Overall, the models demonstrated reasonable predictive capability, with Random Forest's average prediction being approximately $0.81$ stars off (based on RMSE).
* Future work could involve exploring more advanced features or different model architectures to improve prediction accuracy further.

This project demonstrates strong skills in data collection, feature engineering, machine learning model implementation, and robust evaluation techniques.

## Technologies Used

* **Python**
* **Pandas**: For data manipulation and cleaning.
* **Scikit-learn (sklearn)**: For implementing various machine learning regression models (Random Forest, KNN, SVR, Decision Tree, Elastic Net) and evaluation metrics.
* **Matplotlib**: For data visualization.
* **Transformers (Hugging Face)**: For sentiment analysis.
* **Requests**: For interacting with the PlanetTerp API.
* **Jupyter Notebook (Google Colab)**: For interactive analysis and documentation.
