## Introduction
**Sentiment Analysis of Movie Reviews**

This project performs sentiment analysis on IMDb movie reviews using Logistic Regression, achieving 89.24% accuracy. It includes an interactive **Streamlit web app** deployed in Google Colab, allowing users to input movie reviews and receive real-time sentiment predictions (positive or negative).

**Features**
Data Preprocessing: Cleans text data by removing stopwords and applying TF-IDF vectorization using NLTK and Scikit-learn.
Model Training: Trains a Logistic Regression model to classify reviews as positive or negative.
Evaluation: Visualizes model performance with a confusion matrix using Seaborn and Matplotlib.
Web Interface: Deploys a Streamlit app for user-friendly sentiment predictions.
Version Control: Hosted on GitHub with clear documentation and reproducible code.

**Dataset**
Uses the IMDb Dataset of 50K Movie Reviews from Kaggle.
Note: The dataset is not included in this repository due to its size and licensing. Download it from Kaggle and place it in the project directory if running locally.

**Requirements**
To run the project, install the required Python libraries:
pip install -r requirements.txt

The requirements.txt includes:
- pandas==2.2.3
- numpy==2.1.1
- scikit-learn==1.5.2
- nltk==3.9.1
- streamlit==1.39.0
- pyngrok==7.2.0
- seaborn==0.13.2
- matplotlib==3.9.2

**Project Structure**
- sentiment_analysis_with_streamlit.ipynb: Jupyter notebook with the full workflow (training, evaluation, and Streamlit setup) for Google Colab.
- app.py: Streamlit app script for the web interface.
- model.pkl: Saved Logistic Regression model.
- vectorizer.pkl: Saved TF-IDF vectorizer.
- requirements.txt: List of Python dependencies.
- .gitignore: Excludes temporary files and the dataset.
- LICENSE: MIT License for code usage.

## How to Run
**In Google Colab**

1. Open sentiment_analysis_with_streamlit.ipynb in Google Colab.
2. Upload IMDB Dataset.csv to Colab’s /content/sample_data/ directory.
3. Run all cells to:
  Install dependencies.
  Train the model and save model.pkl and vectorizer.pkl.
  Launch the Streamlit app via ngrok.
4. Access the web interface via the ngrok URL (requires an ngrok authtoken).

**Locally**
1. Clone the repository:
git clone https://github.com/<your-username>/sentiment-analysis-movie-reviews.git
2. Install dependencies:
pip install -r requirements.txt
3. Place IMDB Dataset.csv in the project directory (optional for training).
4. Run the Streamlit app:
streamlit run app.py
5. Open the local URL (e.g., http://localhost:8501) in a browser.

# Results
## 2025/7/24
## v1.0.12 Support Vector Machine (SWM)
- Complete 3 test so as to apply XGBoost.

Test 1:
- XGBoost hyperparameter tuning:
- Increase the vectorizer max feature to 15,000 and add trigrams.

- Result: GPU power is not enough to run, disconect.

Test 2:
- Test 1 + enable GPU G4

Result:
50% accurary which like flip a coin, it seems no proper training.

Error:
- /usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py:317: UserWarning: The total space of parameters 8 is smaller than n_iter=10. Running 8 iterations. For exhaustive searches, use GridSearchCV.
  warnings.warn(
- /usr/local/lib/python3.11/dist-packages/xgboost/training.py:183: UserWarning: [04:14:19] WARNING: /workspace/src/common/error_msg.cc:27: The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` parameter to CUDA instead.

    - E.g. tree_method = "hist", device = "cuda"

- bst.update(dtrain, iteration=i, fobj=obj)
  - Best Parameters: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.01}
  - Best Cross-Validation Accuracy: 0.57
- /usr/local/lib/python3.11/dist-packages/xgboost/training.py:183: UserWarning: [04:14:22] WARNING: /workspace/src/common/error_msg.cc:27: The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` parameter to CUDA instead.

    - E.g. tree_method = "hist", device = "cuda"

- bst.update(dtrain, iteration=i, fobj=obj)
- /usr/local/lib/python3.11/dist-packages/xgboost/core.py:2676: UserWarning: [04:14:27] WARNING: /workspace/src/common/error_msg.cc:27: The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` parameter to CUDA instead.

    - E.g. tree_method = "hist", device = "cuda"

  - if len(data.shape) != 1 and self.num_features() != data.shape[1]:
- /usr/local/lib/python3.11/dist-packages/xgboost/core.py:729: UserWarning: [04:14:27] WARNING: /workspace/src/common/error_msg.cc:58: Falling back to prediction using DMatrix due to mismatched devices. This might lead to higher memory usage and slower performance. XGBoost is running on: cuda:0, while the input data is on: cpu.

Potential solutions:
- Use a data structure that matches the device ordinal in the booster.
- Set the device for booster before call to inplace_predict.

This warning will only be shown once.

  return func(**kwargs)
Test Accuracy: 0.50

Test 3:
## v1.0.12 Support Vector Machine (SWM) 
# Add branch Arthuric367-v1.0.13
- Use tree_method='hist' and device='cuda' for proper GPU support.
- Convert TF-IDF output to dense format for GPU compatibility.
- Reduce max_features to 10,000 and limit to bigrams (ngram_range=(1, 2)) to avoid memory issues.
- Expand param_dist for better tuning.
- Keep the subset approach (subset_size=10000) for tuning, then train on the full dataset.

Result:
- Best Parameters: {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.8}
- Best Cross-Validation Accuracy: 0.84
- Test Accuracy: 0.87
- <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/abfd8123-9721-442f-a6b2-c54435915af7" />


## 2025/7/24
## v1.0.12 Support Vector Machine (SWM)
Try to replace for SVC by XGBoost to save resources running on Colab.

- /usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py:317: UserWarning: The total space of parameters 8 is smaller than n_iter=10. Running 8 iterations. For exhaustive searches, use GridSearchCV.
  warnings.warn(
- Best Parameters: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1}
- Best Cross-Validation Accuracy: 0.83
- Test Accuracy: 0.85
- <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/759a1f1e-680b-40db-97e3-66aa73d00108" />

## 2025/7/22
## v1.0.11 Support Vector Machine (SWM)
- Best Parameters: {'C': np.float64(1.5751320499779735), 'gamma': 'scale', 'kernel': 'rbf'}
- Best Cross-Validation Accuracy: 0.88
**Remark: Resources not enough to run SVC**

## v1.0.11 Support Vector Machine (SVM)
Hyperparameter tuning:
- Optimize SVC Tuning with Reduced Parameters
- To make SVC with RBF kernel feasible in Colab:

- Reduce n_iter: Lower the number of iterations in RandomizedSearchCV to 5-10.
- Limit Data Size: Use a smaller subset of the IMDb dataset (e.g., 10,000 reviews) for tuning, then train the final model on the full dataset.
- Narrow Parameter Range: Focus on a smaller C and gamma range to reduce computation.

Result:
- Best Parameters: {'C': np.float64(1.5751320499779735), 'gamma': 'scale', 'kernel': 'rbf'}
- Best Cross-Validation Accuracy: 0.88

## 2025/07/18
## v1.0.11 Support Vector Machine (SVM)
Tune Additional Parameters:
Parameters:
- C: Try a finer grid around 0.1 (e.g., [0.05, 0.1, 0.2, 0.5]) to pinpoint the optimal regularization.
- loss: Test hinge (default) vs. squared_hinge to adjust the loss function.
- dual: Set to True or False (default depends on data size; False is faster for large datasets like yours).
- #Reminder: Either **loss** or **dual** could exist for each time

Runtime: 81.966s - 93s
- Result: 
- Best C: 0.2
- Test Accuracy: 0.90
- Accuracy: 0.89
- <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/05f43c77-837f-4d82-a0b4-3027ebd8eea6" />

Testing with Non-Linear SVM (SVC with Kernel)
- Colab standard Colab T4 GPU not capable of running
- No Respond
  
## 2025/07/17
## v1.0.01 Logistic Regression
1. Logistic Regression with uning the regularization parameter C
- Best C: 1
- Best Cross-Validation Accuracy: 0.89
- Test Accuracy: 0.89
- <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/d365ae0a-386e-49d7-abca-6e20a3f00093" />

2. Try a wider range of C values and include additional Logistic Regression parameters like solver or penalty to find a better configuration.
- Wider C Range: Test smaller and larger values to capture potential improvements.
- Additional Parameters: Include solver (e.g., lbfgs, liblinear) and penalty (e.g., l2, l1 for liblinear).
- Add N-grams:
- **Modify the TfidfVectorizer**:
  -> vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
  - Best C:
  - Best Cross-Validation Accuracy: 0.89
  - Test Accuracy: 0.89
  - <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/d365ae0a-386e-49d7-abca-6e20a3f00093" />

- **Increase max_features:**
- Change to max_features=10000 (from 5000) to include more words.
  - Best C: 1
  - Best Cross-Validation Accuracy: 0.90
  - Test Accuracy: 0.90
  - <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/83edb2d2-8d7b-414f-8873-0c0774291a31" />

## v1.0.10 Support Vector Machine (SVM)
1. Support Vector Machine (SVM)
   - Accuracy: 0.89
   - <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/090541e4-1f11-4355-9848-474821291de4" />

v1.0.10 Support Vector Machine (SVM)
2. Support Vector Machine (SVM) Hyperparameter tuning: Optimal C in LinearSVC
   - Best C: 0.1
   - Test Accuracy: 0.90
   - Accuracy: 0.89
   - <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/cba68a1d-11e6-4345-9a5d-efea49adfbfa" />

## v1.0.20 Naive Bayes
- Accuracy: 0.87
- <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/6c8ffd42-4ae7-4a30-b2ee-e0433bed0d62" />

## v1.0.30 Random Forest
- Accuracy: 0.86
- <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/d2fd543a-519b-45b2-a2f8-0fe5b0e5ac22" />

## v1.0.40 Deep Learning (LSTM)
- 500/500 ━━━━━━━━━━━ 114s 223ms/step - accuracy: 0.7742 - loss: 0.4621 - val_accuracy: 0.8855 - val_loss: 0.2795
  - Epoch 2/5
- 500/500 ━━━━━━━━━━━ 146s 232ms/step - accuracy: 0.9119 - loss: 0.2259 - val_accuracy: 0.8834 - val_loss: 0.2853
  - Epoch 3/5
- 500/500 ━━━━━━━━━━━ 139s 228ms/step - accuracy: 0.9351 - loss: 0.1736 - val_accuracy: 0.8804 - val_loss: 0.3005
  - Epoch 4/5
- 500/500 ━━━━━━━━━━━ 143s 230ms/step - accuracy: 0.9428 - loss: 0.1495 - val_accuracy: 0.8777 - val_loss: 0.3304
  - Epoch 5/5
- 500/500 ━━━━━━━━━━━ 112s 224ms/step - accuracy: 0.9600 - loss: 0.1121 - val_accuracy: 0.8726 - val_loss: 0.3477
- 313/313 ━━━━━━━━━━━ 11s 33ms/step - accuracy: 0.8758 - loss: 0.3478
- Accuracy: 0.88
- 313/313 ━━━━━━━━━━━ 10s 32ms/step
- <img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/abcbe47a-6325-4ab7-8773-2c58ddbc2c5f" />
- 1/1 ━━━━━━━━━━━━━━━ 0s 51ms/step

# 2025/07/16
## v1.0.00 Logistic Regression
Accuracy: 89.24% on the test set.
Example Prediction:
- Input: "This movie was amazing!"
- Output: Predicted Sentiment: positive
Visualization: Includes a confusion matrix to show model performance.
<img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/ffa05351-1a47-438c-835d-bad108744ef9" />


**License**
This project is licensed under the MIT License. See the LICENSE file for details.

**Acknowledgments**
Built with Scikit-learn, NLTK, and Streamlit.
Dataset sourced from Kaggle.
Web app deployed using ngrok in Google Colab.


Dataset sourced from Kaggle.



Web app deployed using ngrok in Google Colab.
