

```markdown
# Chatbot Implementation project

## AIM
To develop a chatbot using Natural Language Processing (NLP) and a Naive Bayes classifier for intent classification. The chatbot takes user input, predicts the intent, and generates an appropriate response based on predefined intents and responses stored in a CSV file.

## DATASET LINK
[https://drive.google.com/file/d/1J7mGS16EkgCEtN7UJtBlJACeqoDbdS4F/view?usp=drive_link]

## NOTEBOOK LINK
[https://colab.research.google.com/drive/1L2LKfbVv4pb4yzczcRnnU4AkEW-kCZSZ?usp=sharing]

## LIBRARIES NEEDED

nltk
scikit-learn
numpy
pickle
csv


## DESCRIPTION

???What is the requirement of the project?
A chatbot is required to automate conversations and provide immediate responses to user queries. It can be used to answer FAQs, provide customer support, and improve user interaction.

??? Why is it necessary?
- Chatbots are essential for improving user engagement and providing 24/7 service.
- They automate responses, saving time and providing immediate help.

??? How is it beneficial and used?
- Chatbots can be used for customer service automation, answering user questions, and guiding users through processes on websites or apps.

### Initial Thoughts and Planning
- Gathered intents and responses in CSV format for training.
- Preprocessed text (tokenization, lemmatization) to prepare it for the model.
- Built a Naive Bayes classifier to predict intents.
- Deployed the model to predict user queries and return appropriate responses.

### Additional Resources Used
- [Scikit-learn Documentation](https://scikit-learn.org)
- Tutorial: Building Chatbots with NLP and Machine Learning



### Features in the Dataset

| Feature   | Description                                       |
|-----------|---------------------------------------------------|
| `intents` | User query categories like greetings, farewells.  |
| `responses`| Predefined chatbot responses for each intent.    |

### Steps and Implementation

1. **Data Preprocessing:**
    - Loaded the intents from CSV files.
    -  Clean data: remove duplicates and handle nulls


2. **Vectorization:**
    - Used `TfidfVectorizer` to convert text into vectors
    - Split data into training and testing sets.

3. **Model Training:**
    - Trained a Naive Bayes classifier on the preprocessed data.
    - Saved the model for future use with `pickle`.
    -Create intent-response mapping


4. **Prediction and Response Generation:**
    - The chatbot predicts the intent based on user input.
    - The appropriate response is fetched and returned.
5.  **Test the chatbot:**
     - Live Interaction with chatbot.



### Features Not Implemented Yet
- Integration of a deep learning model (e.g., RNN or LSTM) for better context handling.
  


Example chatbot interaction:

```text
you: Hello
Bot: Hi, How I can assist you?
```

---

## MODELS AND EVALUATION METRICS

| Model            | Accuracy | Precision | Recall |
|------------------|----------|-----------|--------|
| Naive Bayes      | 92%      | 91%       | 90%    |

---

## FUTURE ENHANCEMENTS
- Implement context handling with more advanced models like LSTM.
- Integrate the chatbot into a web interface for live user interactions.


## CONCLUSION

### What have you learned?
- Building a chatbot using NLP techniques can automate interactions and provide user-friendly interfaces for businesses.
- The Naive Bayes classifier is an effective yet simple model for intent prediction.

### Use Cases of this Model:
1. Customer Support Automation: Provide 24/7 automated support for customers.
2. FAQ Automation: Automatically respond to frequently asked questions on websites or apps.
   
### Screenshots ###
![Screenshot](https://github.com/salmasyed1909/chatb/blob/main/Screenshot%20(255).png)

###Flow Chart###
```mermaid
graph LR
    A[Start] --> B[User Input];
    B --> C[Text Preprocessing];
    C --> D[Intent Classification];
    D -->|Greeting| E[Greeting Intent];
    D -->|Question| F[Question Intent];
    D -->|Command| G[Command Intent];
    D -->|Fallback| H[Fallback Intent];
    E --> I[Send Greeting Response];
    F --> J[Answer Question];
    G --> K[Execute Command];
    H --> L[Ask for Clarification];
    I --> M[End];
    J --> M;
    K --> M;
    L --> B;
'''
