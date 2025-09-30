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

## 📊 Exploratory Data Analysis (EDA)

Key insights were drawn from visualizing the data distributions.

| Engagement Distribution | User Age Distribution |
| :---: | :---: |
| <img src="https://i.imgur.com/r6E8c9H.png" alt="Engagement Distribution Plot" width="400"/> | <img src="https://i.imgur.com/qR8b9mU.png" alt="User Age Distribution Plot" width="400"/> |
| **Observation**: The target variable `engagement` is fairly balanced, with slightly more non-engagements (0) than engagements (1). | **Observation**: The user base is primarily young, with most users between the ages of 20 and 35. |

| Gender Distribution | Content Type Distribution |
| :---: | :---: |
| <img src="https://i.imgur.com/J3t5Xv8.png" alt="Gender Distribution Plot" width="400"/> | <img src="https://i.imgur.com/gq5c9gM.png" alt="Content Type Distribution Plot" width="400"/> |
| **Observation**: The dataset has a relatively even distribution across Female, Male, and Other gender identities. | **Observation**: Posts are distributed across three content types: video, image, and text. |

---

## 🛠️ Methodology & Feature Engineering

### Feature Engineering

A key feature, `interest_match`, was created to measure the alignment between a user's interests and a post's tags. It is calculated as the count of common elements between a user's `top_3_interests` and the post's `tags`.

```python
# Feature: Interest match count
merged_df['interest_match'] = merged_df.apply(
    lambda row: len(set(row['top_3_interests']) & set(row['tags'])), axis=1
)
