# Social Media Engagement Prediction 🚀

This project builds a machine learning model to predict whether a user will engage with a social media post. The prediction is based on user demographics, their interests, the type of post, and its content tags. It serves as a complete walkthrough of a binary classification problem, from data exploration to model evaluation.

---

## 📋 Project Overview

The core task is a binary classification problem: predicting engagement (`1`) or no engagement (`0`). The workflow involves:
1.  **Data Loading & Inspection**: Loading and checking the quality of three separate datasets (`Users.csv`, `Posts.csv`, `Engagements.csv`).
2.  **Exploratory Data Analysis (EDA)**: Visualizing the data to understand distributions and relationships.
3.  **Feature Engineering**: Creating a new feature, `interest_match`, to quantify the overlap between user interests and post tags.
4.  **Data Preprocessing**: Merging the datasets and preparing the features for modeling using `StandardScaler` for numerical data and `OneHotEncoder`/`MultiLabelBinarizer` for categorical data.
5.  **Model Training**: Building and training a Logistic Regression model.
6.  **Evaluation**: Assessing the model's performance using various classification metrics like Accuracy, ROC AUC, and a confusion matrix.

---

## 💾 Dataset

The data is provided in three separate CSV files:
* `Users.csv`: Contains user information like `user_id`, `age`, `gender`, `top_3_interests`, and `past_engagement_score`.
* `Posts.csv`: Contains post information like `post_id`, `creator_id`, `content_type`, and `tags`.
* `Engagements.csv`: This is the target dataset, mapping `user_id` and `post_id` to an `engagement` outcome (0 or 1).

---

---

## 🛠️ Methodology & Feature Engineering

### Feature Engineering

A key feature, `interest_match`, was created to measure the alignment between a user's interests and a post's tags. It is calculated as the count of common elements between a user's `top_3_interests` and the post's `tags`.

# Feature: Interest match count
merged_df['interest_match'] = merged_df.apply(
    lambda row: len(set(row['top_3_interests']) & set(row['tags'])), axis=1
)


### Preprocessing Pipeline
A `ColumnTransformer` from scikit-learn was used to apply different preprocessing steps to different types of columns in an organized pipeline:

* **Numerical Features** (`age`, `past_engagement_score`, `interest_match`): Scaled using `StandardScaler`.
* **Categorical Features** (`gender`, `content_type`): Encoded using `OneHotEncoder`.
* **Multi-Label Features** (`top_3_interests`, `tags`): Transformed into binary columns using `MultiLabelBinarizer` and passed through without further scaling.

---

### 📈 Model Performance
A **Logistic Regression** model was trained on the preprocessed data. The `class_weight='balanced'` parameter was used to handle the slight class imbalance.

The model's performance on the test set is summarized below:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.485 |
| **ROC AUC Score** | 0.488 |
| **Precision (Class 1)**| 0.48 |
| **Recall (Class 1)** | 0.45 |
| **F1-Score (Class 1)** | 0.47 |


---

### 💡 Conclusion & Next Steps
The initial Logistic Regression model achieved an accuracy of **48.5%** and a ROC AUC of **0.488**. This performance is slightly below the 0.5 baseline of a random guess, indicating that the current features and model are not sufficient to effectively predict user engagement.

This project successfully establishes a baseline. To improve performance, future work could include:

* **Exploring More Complex Models**: Algorithms like Gradient Boosting (XGBoost, LightGBM) or Random Forest might capture more complex patterns in the data.
* **Advanced Feature Engineering**: Create more interaction features, such as the popularity of a post's creator or time-based features if timestamps were available.
* **Hyperparameter Tuning**: Optimize the model's parameters using techniques like GridSearchCV or RandomizedSearchCV.
* **Collect More Data**: A larger dataset with more diverse features could significantly enhance the model's predictive power.

---

### ⚙️ How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
    

3.  **Place the data files** (`Users.csv`, `Posts.csv`, `Engagements.csv`) in the root directory of the project.

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Assesment_test.ipynb
    ```
