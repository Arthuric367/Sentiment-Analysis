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
pandas==2.2.3
numpy==2.1.1
scikit-learn==1.5.2
nltk==3.9.1
streamlit==1.39.0
pyngrok==7.2.0
seaborn==0.13.2
matplotlib==3.9.2

**Project Structure**
sentiment_analysis_with_streamlit.ipynb: Jupyter notebook with the full workflow (training, evaluation, and Streamlit setup) for Google Colab.
app.py: Streamlit app script for the web interface.
model.pkl: Saved Logistic Regression model.
vectorizer.pkl: Saved TF-IDF vectorizer.
requirements.txt: List of Python dependencies.
.gitignore: Excludes temporary files and the dataset.
LICENSE: MIT License for code usage.

**How to Run**
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

**Results**
Accuracy: 89.24% on the test set.
Example Prediction:
  Input: "This movie was amazing!"
  Output: Predicted Sentiment: positive
Visualization: Includes a confusion matrix to show model performance.

**License**
This project is licensed under the MIT License. See the LICENSE file for details.

**Acknowledgments**
Built with Scikit-learn, NLTK, and Streamlit.
Dataset sourced from Kaggle.
Web app deployed using ngrok in Google Colab.


Dataset sourced from Kaggle.



Web app deployed using ngrok in Google Colab.
