# Project Title : Healthcare Chatbot Using Machine Learning and Natural Language Preocessing
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
Here’s a list of all the datasets used in your project:

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
   - Gather multiple healthcare-related datasets from public sources like Kaggle and official health portals:  
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
├── chest_xray/
│   │  ├──test/
│   │  │  ├──NORMAL/
│   │  │  └──PENUMONIA/
│   │  ├──train/
│   │  │  ├──NORMAL/
│   │  │  └──PNEUMONIA/
│   │  ├──val/
│   │  │  ├──NORMAL/
│   │  │  └──PNEUMONIA/
├── Images/
│   │  ├──category_distribution_bar.png 
│   │  ├──category_distribution_pie.png
│   │  ├──confusion_matrix.png
│   │  └──image_display.png
├── saved_models/
│   │  ├──best_model.h5  
├── static/
│   │   ├── css/
│   │   ├──js/
│   │   ├──img.jpeg
│   │   └──main.jpeg
├── templates/
│   │   ├──result.html
│   │   └── index.html
├── uploads/   
├── app.py
├── classification_report.txt
├── Dataset_Visualization.ipynb
├── models_results.txt
├── pneumonia_detection_transfer_learning.ipynb
└── README.md
```
## Key files
- **`Pneumonia_Detection/templates/index.html`**: The main HTML file the application interface
- **`Pneumonia_Detection/app.py`**:The app.py file typically contains the main Flask application logic, including route definitions, model loading, and image prediction handling for the pneumonia detection system.
- **`Pneumonia_Detection/pneumonia_detection_transfer_leraning.ipynb`**: The file `pneumonia_detection_transfer_learning (2).ipynb` contains a transfer learning-based implementation using a pretrained model (likely VGG16 and EfficientNet-B0) to classify chest X-ray images into Pneumonia and Normal categories, enhancing performance with reduced training time.
## Project Ouput Images
1. Home page of the application [here](https://drive.google.com/file/d/1d9znJ6vk63_HBbLtDFe2XrOK8lHLHJNn/view?usp=drive_link)
2. Pneumonia chest x-ray image uploaded [here](https://drive.google.com/file/d/1RKhDq7klpBsSw-S7kbKn7lrr3iLF6VdJ/view?usp=drive_link)
3. Image classified as Pneumonia [here](https://drive.google.com/file/d/1O6Q3gRiKHu0WyFCtso253gpseN38pWjL/view?usp=drive_link)
4. Normal chest x-ray image uploaded [here](https://drive.google.com/file/d/1Bz0vsnS2mlNT7hIEgIjywYf9D5zwNtrX/view?usp=drive_link)
5. Image classified as Normal [here](https://drive.google.com/file/d/1chQDDWYuB22FavN1DZGZ5Bz0oxhwhTSs/view?usp=drive_link)
## Project Execution Video
For a detailed demonstration of Pneumonia Detection, you can watch the project video here [here](https://drive.google.com/file/d/1gTRoixlaL1WpGmOQ0CuSCidXgXaTJmxj/view?usp=drive_link)
   
