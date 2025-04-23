# Project Title : Healthcare Chatbot Using Machine Learning and Natural Language Processing
The aim of this project is to develop an intelligent healthcare assistant that predicts diseases based on user symptoms and provides personalized recommendations. It offers disease descriptions, precautions, medications, workouts, diet plans, and connects users to relevant specialists.

## Features
- Symptom-based Disease Prediction

- Healthcare Recommendations

- Doctor Suggestion System

- User-Friendly Chatbot Interface

- Ensemble Learning Techniques
- 
## Tech Stack
- Backend: Python, Flask
- Frontend: Html, CSS, Javascript
- Models: KNN, Adaboost ,MLP

## Prerequisites
- Python 3.11 or higher
- Flask
- Anaconda
- Jupyter notebook
  
## Dataset
Here’s a list of all the datasets used in the project:

1. **Symptoms-Disease Dataset**  
   Contains mappings of various symptoms to corresponding diseases, used for training the machine learning models for disease prediction.

2. **Disease Description Dataset**  
   Provides detailed information about each disease to help users understand the nature and causes of the illness.

3. **Precautions Dataset**  
   Lists preventive measures and do’s & don’ts for each disease to help users manage or avoid complications.

4. **Medication Dataset**  
   Includes common medicines and treatments associated with each disease, offering users basic guidance on possible remedies.

5. **Exercise Dataset**  
   Suggests physical activities suitable for patients based on their diagnosed disease, promoting recovery and well-being.

6. **Diet Plan Dataset**  
   Contains nutritional recommendations tailored to specific diseases to support users with healthy eating habits.

7. **Doctors Dataset**  
   Maps diseases to specialized doctors based on medical domains, helping users connect with the right healthcare professionals.

  

## Step by step procedure for model development 

### 1. **Problem Definition**  
   - **Identify the goal**: Predict the disease based on user-reported symptoms using machine learning and provide relevant healthcare recommendations.



### 2. **Data Collection**  
   - Gathered multiple healthcare-related datasets from public sources like Kaggle and official health portals:  
     - Symptoms-Disease mappings  
     - Disease descriptions  
     - Precautions  
     - Medications  
     - Workouts and diet plans  
     - Doctor specializations  

### 3. **Data Preprocessing**  
   - Clean and format all datasets  
   - Encode symptoms and diseases  
   - Split data into features (symptoms) and target (disease)  
   - Apply label encoding and train-test split  


### 4. **Model Building**  
   - Train multiple machine learning models: KNN, AdaBoost, MLP  
   - Use **ensemble learning** (majority voting) to combine predictions from top-performing models  


### 5. **Model Compilation**  
   - Define evaluation metrics (accuracy, precision, recall)  
   - Choose suitable hyperparameters for each model  

### 6. **Model Training**  
   - Train models on the symptom-disease dataset  
   - Validate using a separate holdout set and tune for best performance  


### 7. **Model Evaluation**  
   - Compare performance of models using accuracy and confusion matrix  
   - Select top 3 models for ensemble based on evaluation metrics  


### 8. **Model Saving**  
   - Save trained models using `joblib` or `pickle` for deployment  
   - Save final ensemble model for real-time predictions  

### 9. **Deployment**  
   - Develop a **Flask-based web application** with a chatbot interface  
   - Allow users to input symptoms and receive disease predictions with full recommendations (precautions, medications, diet, etc.)  


## Usage
1. **Run the Flask Application**
   ```bash
   flask run
   ```
2. **Access the webpage**
   Open the browser and go to `http://127.0.0.1:5000` to use the web application
## Project Structure
```
PROJECT_CHAT_BOT/
├── Medical_Dataset/
│   │  ├──Kaggle_dataset/
│   │  │  ├──description/
|   |  |  └──symptom_severity/
|   |  |  └──symptom_des/
|   |  |  └──precautions/
|   |  |  └──diets/
|   |  |  └──workouts/
|   |  |  └──disease_specializations/
|   |  |  └──Indian_doctors_dataset/
│   │  ├──full dataset/
│   │  ├──Training/
│   │  ├──Testing/
├── TRAINED_MODELS/
│   │  ├──m_v.pkl  
├── static/
│   │  ├──styles/
│   │  |    ├── style.css/
│   │  |    └──back.png
├── templates/
│   │   ├──home.html
│   │   ├──bmi.html
│   │   ├──medicine.html
│   │   ├──mental.html
│   │   └──goals.html  
├── app.py
├── HCC.ipynb
├── DATA.json
└── README.md
```
## Key files
- **`PROJECT_CHAT_BOT/templates/home.html`**: The main HTML file for the chatbot interface, serving as the landing page for users.
- **`PROJECT_CHAT_BOT/app.py`**:The app.py file typically contains the main Flask application logic, including route definitions, model loading, and symptom input handling, and disease prediction using the trained ensemble model.
- **`PROJECT_CHAT_BOT/HCC.ipynb`**: he Jupyter Notebook used for model development, including data preprocessing, training, evaluation, and ensemble implementation using KNN, AdaBoost, and MLP classifiers.
- - **`PROJECT_CHAT_BOT/TRAINED_MODELS/m_v.pkl`**: The trained machine learning model file saved in pickle format. It performs majority voting based on predictions from KNN, AdaBoost, and MLP.

## Project Ouput Images
1. Home page of the application [here](https://drive.google.com/file/d/1roGZ7yuZqUHXYpHfXEhxt7sza6oMvmBf/view?usp=sharing)
2. Symptom Inquiry process by chabbot [here](https://drive.google.com/file/d/1ozWYQrHGEC0Onn3glQ8hwjKoXy5czuto/view?usp=sharing)
3. Disease Predition and discription by chatbot  [here](https://drive.google.com/file/d/14wcEORa4t_SqgthxcCr3GSs7FkTwMD8y/view?usp=sharing)
4. Precautions by chatbot [here](https://drive.google.com/file/d/1EJuKvm_fqnpqy1zVP6vT-frOOjXwkhnM/view?usp=sharing)
5. Diet suggestions by chatbot [here](https://drive.google.com/file/d/1vg-OqJf9okorQU-X60KqEXfIBbLqhzHV/view?usp=sharing)
6. Workout and medical specialist recommendations by chatbot [here](https://drive.google.com/file/d/1tCqQMbn6k0YRGiL4j-ZLakYPq_iGI27y/view?usp=sharing)
7. Doctor referal by chatbot [here](https://drive.google.com/file/d/1tCqQMbn6k0YRGiL4j-ZLakYPq_iGI27y/view?usp=sharing)
8. Diagnosis summary by chabot[here](https://drive.google.com/file/d/1NZQj-uLJc5zYljpnu34qIcWCy-f1nDlc/view?usp=sharing)
   

   
