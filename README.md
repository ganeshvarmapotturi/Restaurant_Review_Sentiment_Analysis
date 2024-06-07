# Restaurant-review-analysis

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Deep Learning Model](#deep-learning-model)
- [Prediction Example](#prediction-example)
- [Future Work](#future-work)
- [Requirements](#requirements)
- [Usage](#usage)

## Project Overview
This project aims to classify restaurant reviews as positive or negative using machine learning and deep learning techniques. The goal is to help restaurant owners and managers quickly understand customer feedback and make informed decisions to improve their services.

## Dataset
The dataset consists of restaurant reviews with corresponding ratings. Ratings are converted to binary labels:
- Ratings below 4 are considered negative (0).
- Ratings of 4 or above are considered positive (1).

## Data Preprocessing
The data preprocessing steps include:
1. Cleaning the text by removing non-alphabetic characters and converting the text to lowercase.
2. Removing stop words.
3. Lemmatizing the text to reduce words to their base form.

## Feature Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is used to transform the text data into numerical features. This helps highlight important words in the reviews by balancing their frequency in the document against their frequency in the entire corpus.

## Model Training
Several machine learning models were trained on the TF-IDF features:
- Naive Bayes
- Logistic Regression
- Random Forest
- Bagging Classifier

The accuracy of these models was evaluated to determine the best performer.

## Model Evaluation
Among the traditional machine learning models, the Logistic Regression model performed the best with a high accuracy score.

## Deep Learning Model
An LSTM (Long Short-Term Memory) model was developed, which is a type of Recurrent Neural Network (RNN) particularly effective for sequential data like text. The model was trained on padded sequences of the reviews, ensuring uniform input size.

## Prediction Example
To demonstrate, the LSTM model was used to predict the sentiment of a sample review:
```python
sample_review = "Its a very nice place, ambience is different, all the food we ordered was very tasty, service is also gud, worth visit. Its reasonable as well. Really a must visit place."
prediction = predict_sentiment(sample_review, tokenizer, MAX_LEN)
print(prediction)
```
The model predicted a positive sentiment, which aligns with the review's tone.

## Future Work
Future work could include expanding the dataset, exploring more advanced models, and incorporating additional features like customer metadata to further enhance predictions.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Seaborn
- Matplotlib
- NLTK
- Scikit-learn
- Keras
- TensorFlow

## Usage
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using:
    ```bash
    pip install -r requirements.txt
    ```
3. Load and preprocess the dataset:
    ```python
    reviews = pd.read_csv('Restaurant reviews.csv')
    reviews_txt = reviews[['Review', 'Rating']]
    reviews_txt['Rating'] = np.where(reviews_txt['Rating']<4, 0, 1)
    ```
4. Clean and preprocess the text data:
    ```python
    def clean(sentence): 
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = cleanup_re.sub(' ', sentence).strip()
        return sentence

    reviews_txt['Review'] = reviews_txt['Review'].apply(clean)

    def preprocess(sentence):
        sentence = str(sentence)
        word_tokens = word_tokenize(sentence)
        stop_words = set(stopwords.words('english'))
        sentence = ' '.join([i for i in word_tokens if not i in stop_words])
        return sentence

    reviews_txt['Review'] = reviews_txt['Review'].apply(preprocess)
    ```
5. Perform feature extraction using TF-IDF:
    ```python
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(X_train)
    xtrain_tfidf =  tfidf_vect.transform(X_train)
    xtest_tfidf =  tfidf_vect.transform(X_test)
    ```
6. Train and evaluate machine learning models:
    ```python
    def train_model(classifier, feature_vector_train, label, feature_vector_valid):
        classifier.fit(feature_vector_train, label)
        predictions = classifier.predict(feature_vector_valid)
        return accuracy_score(predictions, y_test)

    accuracy = train_model(MultinomialNB(), xtrain_tfidf, y_train, xtest_tfidf)
    print("NB, WordLevel TF-IDF: ", accuracy)
    ```
7. Train and evaluate the LSTM model:
    ```python
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim= EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(LSTM(300, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64)
    model.save("LSTM.h5")
    ```
8. Use the trained model for prediction:
    ```python
    loaded_model = load_model("LSTM.h5")
    prediction = predict_sentiment(sample_review, tokenizer, MAX_LEN)
    print(prediction)
    ```

---

