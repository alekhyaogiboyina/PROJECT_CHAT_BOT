import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import csv
import json
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib
from flask import Flask, render_template, request, session
import ast
import nltk
from sklearn.model_selection import train_test_split
import os
import re
from flask import Flask, jsonify, request, send_file, session
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

#Load the pre-trained spaCy model for natural language processing (NLP)
nlp = spacy.load('en_core_web_sm')

# save data
data = {"users": []}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)

def write_json(new_data, filename='DATA.json'):
    with open(filename, 'r+') as file:
        # Load existing data into a dict
        file_data = json.load(file)
        
        # Check if new_data is already in the "users" list
        if not any(user == new_data for user in file_data["users"]):
            # Append new_data only if it's not a duplicate
            file_data["users"].append(new_data)
        
        # Move file's current position to the beginning
        file.seek(0)
        
        # Convert back to JSON and overwrite file content
        json.dump(file_data, file, indent=4)
        # Truncate any remaining content in case of shorter new data
        file.truncate()

JSON_FILE = "DATA.json"

df_tr = pd.read_csv(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\Training.csv')
df_tt=pd.read_csv(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\Testing.csv')


symp = []  # List to store symptoms
disease = [] # List to store disease
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list()) #select the columns where the value is 1 (indicating the symptom is present),convert the column names with a value of 1 to a list of symptoms
    disease.append(df_tr.iloc[i, -1]) #append the disease name for the current row to the disease list.


all_symp_col = list(df_tr.columns[:-1])  #List of symptom column names (all except the last column)

# Replace underscores with spaces, remove unwanted substrings, and standardize certain symptom terms

def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace(
        'yellowing', 'yellow').replace('sneezing','sneeze')

# Apply the clean_symp function to all symptom column names to clean them
all_symp = [clean_symp(sym) for sym in (all_symp_col)]


def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        # Avoid stop words(the,is,have........) and non-alphabetical tokens(1,special characters)
        if not token.text.lower() in STOP_WORDS and token.text.isalpha():
            # Check for compound terms like 'abdominal_pain' and keep them intact
            if "_" in token.text:
                d.append(token.text.lower())  # Preserve compound words as they are
            else:
                d.append(token.lemma_.lower())  # Lemmatize other words
    return ' '.join(d)

all_symp_pr = [preprocess(sym) for sym in all_symp] #list that has all preprocessed symp

# associate each processed symp with original symptomn column name
col_dict = dict(zip(all_symp_pr, all_symp_col))


# Returns all the subsets of a set. This is a generator.
# {1,2,3}->[{},{1},{2},{3},{1,3},{1,2},..]
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item

# Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a

# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])

# check if a txt and all diferrent combination if it exists in processed symp list
def DoesExist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        for sym in permutations(comb):
            if sym in all_symp_pr:
                return sym
    return False

# Jaccard similarity 2docs
def jaccard_set(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

# apply vanilla jaccard to symp with all corpus
def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
       if symp_t.lower() in symp.lower():  # Case insensitive substring match
            poss_sym.append(symp)  # Add the symptom to the possible symptoms list
            most_sim.append(1)
       else:
            d = jaccard_set(symp_t, symp)  # Use Jaccard similarity for other cases
            most_sim.append(d)
    # Sort by most similar symptoms
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if most_sim[i] != 0 and corpus[i] not in poss_sym:  # Avoid adding duplicate symptoms
            poss_sym.append(corpus[i])
    if len(poss_sym)>=1:
        return 1, poss_sym
    else:
        return 0, None


# check a pattern if it exists in processed symp list
def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None


from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


def WSD(word, context):
    sens = lesk(context, word)
    return sens


# semantic similarity 2docs
def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                # x=syn1.path_similarity((syn2))
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


# apply semantic simarity to symp with all corpus
def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


# given a symp suggest possible synonyms
def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))


# One-Hot-Vector dataframe
def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)

def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a

# list of symptoms --> possible diseases
def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis

# disease --> all symptoms
def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

# print possible symptoms
def related_sym(psym1):
    s = "could you be more specific, <br>"
    i = len(s)
    for num, it in enumerate(psym1):
        s += str(num) + ") " + clean_symp(it) + "<br>"
    if num != 0:
        s += "Select the one you meant."
        return s
    else:
        return 0


model= joblib.load(r'D:\PROJECT_CHAT_BOT\TRAINED_MODELS\m_v.pkl')


severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
medicationDictionary = dict()
exerciseSuggestionsDictionary = dict()
dietDictionary = dict()
specialistDictionary = dict()
hospitalDoctorDictionary = dict()
symptomDescriptionDictionary =dict()

def getHospitalDoctorDict():
    global hospitalDoctorDictionary
    with open(r'D:\HEALTHCARE_CHATBOT_PROJECT\Disease-Symptom-Prediction-Chatbot\Medical_dataset\Indian_doctors_dataset.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            # Extract relevant data from each row
            name = row[0]
            specialization = row[1]
            city = row[2].lower()  # Normalize city to lowercase
            hospital = row[3]
            experience = row[4]

            # Create a key based on specialization and city
            key = (specialization, city)
            
            # Add doctor details under the respective key
            if key not in hospitalDoctorDictionary:
                hospitalDoctorDictionary[key] = []
            
            hospitalDoctorDictionary[key].append({
                "name": name,
                "hospital": hospital,
                "experience": experience,
            })

def getSpecialistDict():
    global specialistDictionary
    with open(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\kaggle_dataset\disease_specializations.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # Each row contains a disease name and a specialist
            _specialist = {row[0]: row[1]}  # Disease name as key, specialist as value
            specialistDictionary.update(_specialist)  # Update the global dictionary with the new entry



def getExerciseSuggestionsDict():
    global exerciseSuggestionsDictionary
    with open(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\kaggle_dataset\workout_df.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # Exercise suggestions are in columns 1 to N (row[1:] represents them)
            _exercise = {row[0]: row[1:]}  # Disease name as key, the rest as exercise suggestions
            exerciseSuggestionsDictionary.update(_exercise)  # Update the global dictionary with the new entry

def getMedicationDict():
    global medicationDictionary
    with open(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\kaggle_dataset\meds.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # Medication suggestions are in column 1 (row[1] contains the medications)
            _medications = {row[0]: row[1:]}  # Disease name as key, medications as a single string
            medicationDictionary.update(_medications)  # Update the global dictionary with the new entry


def getDietDict():
    global dietDictionary
    with open(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\kaggle_dataset\diets.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # Diet suggestions are in columns 1 to N (row[1:] represents them)
            _diet = {row[0]: row[1:]}  # Disease name as key, the rest as diet suggestions
            dietDictionary.update(_diet)  # Update the global dictionary with the new entry

def getDescription():
    global description_list
    with open(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\kaggle_dataset\description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\kaggle_dataset\symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open(r'D:\PROJECT_CHAT_BOT\Medical_Dataset\kaggle_dataset\precautions_df.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

def getSymptomDescriptionDict():
    global symptomDescriptionDictionary

    with open(r'D:\HEALTHCARE_CHATBOT_PROJECT\Disease-Symptom-Prediction-Chatbot\Medical_dataset\kaggle_dataset\Symp_des.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                if len(row) >= 2:  # Ensure there are at least two columns
                    _diction = {row[0].strip(): row[1].strip()}
                    symptomDescriptionDictionary.update(_diction)
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")


# load dictionaries
getSeverityDict()
getprecautionDict()
getDescription()
getDietDict()
getMedicationDict()
getExerciseSuggestionsDict()
getSpecialistDict()
getHospitalDoctorDict()
getSymptomDescriptionDict()

# calcul patient condition
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary.keys():
            sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp)) > 13):
        return 1 
    else:
        return 0

@app.route("/")
def home():
    return render_template('home.html')


@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

@app.route('/medicine')
def medicine():
    return render_template('medicine.html')

@app.route('/mental')
def mental():
    return render_template('mental.html')

@app.route('/goals')
def goals():
    return render_template('goals.html')

@app.route("/download_pdf", methods=["GET"])
def download_pdf():
    try:
        # Load JSON data
        with open(JSON_FILE, "r") as file:
            data = json.load(file)

        if not data or "users" not in data or not data["users"]:
            return jsonify({"warning": "Diagnosis is still in progress. Please wait until it is completed before printing."}), 400
        
        # Create PDF
        pdf_file = r"D:\PROJECT_CHAT_BOT\reccomendations.pdf"
        doc = SimpleDocTemplate(pdf_file)
        styles = getSampleStyleSheet()
        elements = []

        title_style = styles["Title"]
        title_style.fontSize = 18
        title_style.leading = 24
        elements.append(Paragraph("Diagnosis Summary", title_style))
        elements.append(Spacer(1, 24))  # Add space after the title

        for entry in data["users"]:
            elements.append(Paragraph(f"<b>Name:</b> {entry.get('Name','N/A')}",styles["BodyText"]))
            elements.append(Paragraph(f"<b>Age:</b> {entry.get('Age','N/A')}",styles["BodyText"]))
            elements.append(Paragraph(f"<b>Gender:</b> {entry.get('Gender','N/A')}",styles["BodyText"]))
            elements.append(Paragraph(f"<b>Disease:</b> {entry.get('Disease', 'N/A')}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Description:</b> {entry.get('Description', 'N/A')}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Precautions:</b> {', '.join(entry.get('Precautions', []))}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Diet Suggestions:</b> {', '.join(entry.get('DietSuggestions', []))}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Medication Suggestions:</b> {', '.join(entry.get('MedicationSuggestions', []))}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Exercise Suggestions:</b> {', '.join(entry.get('ExerciseSuggestions', []))}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Specialist Recommended:</b> {entry.get('Specialist', 'N/A')}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Location:</b> {entry.get('Location', 'N/A')}", styles["BodyText"]))
            
            doctors = entry.get("Doctors Recommended", [])
            if doctors:
                elements.append(Paragraph("<b>Doctors Recommended:</b>", styles["BodyText"]))
                for doctor in doctors:
                    elements.append(Paragraph(f"- {doctor['name']} at {doctor['hospital']} (Experience: {doctor['experience']} years)", styles["BodyText"]))
            else:
                elements.append(Paragraph("<b>Doctors Recommended:</b> NONE RECOMMENDED", styles["BodyText"]))

            elements.append(Spacer(1, 12)) 
        # Build and save the PDF
        doc.build(elements)

        # Serve the PDF for download
        return send_file(pdf_file, as_attachment=True,download_name="Diagnosis.pdf")

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/get")
def get_bot_response():
    s = request.args.get('msg')
    if "step" in session:
        if session["step"] == "Q_C":
            name = session["name"]
            age = session["age"]
            gender = session["gender"]
            session.clear()
            if s == "q":
                "Thank you for using ower web site" + name
            else:
                session["step"] = "FS"
                session["name"] = name
                session["age"] = age
                session["gender"] = gender

    if 'step' not in session:
        if s.upper() in ["OK", "OKAY", "SURE", "YES", "ALRIGHT", "YES, PLEASE", "GO AHEAD", "PROCEED", "LET'S GO", "I'M READY", "AFFIRMATIVE"]:
            session['step'] = "name"
            return "What is your name?"
        else:
            return "I would like to have your Consent !Please type OK to begin."

    # Ask for the user's name if the step is "name"
    if session.get("step") == "name":
        if s.replace(" ", "").isalpha():  # Check if the name consists of only letters and spaces
            session['name'] = s
            session['step'] = "age"
            return "How old are you?"
        else:
            return "Please enter a valid name (letters only)."

    # Handle the age input
    if session.get("step") == "age":
        try:
            session["age"] = int(s)
            session["step"] = "gender"
            return "Can you specify your gender(M/F)?"
        except ValueError:
            return "Please enter your age in numbers."
        
    if session["step"] == "gender":
        valid_genders = ["M", "m", "Male", "male", "F", "f", "Female", "female"]
        if s not in valid_genders:
            return "Please specify your gender properly. You can enter M for Male or F for Female."
        session["gender"] = s
        session["step"] = "Depart"

        if s.lower() in ["m", "male","Male"]:
            session["title"] = "Mr."
        elif s.lower() in ["f", "female","Female"]:
            session["title"] = "Ms."
            
    if session['step'] == "Depart":
        session['step'] = "BFS" #begin first Sympt
        return "Well, Hello again, "+session["title"] + session["name"]+" ,now I will be asking some few questions about your symptoms to see what you should do. Type S to start diagnosis!"
    if session['step'] == "BFS":
        if s.lower() == "s":  # If the user taps S to start diagnostic
            session['step'] = "FS"  # Move to First Symptom step
            return "Can you precise your main symptom "+ session["title"] + session["name"] + " ?"
        else:
            return "enter s"
    if session['step'] == "FS":
        sym1 = s # Capture the user's input for the first symptom
        sym1 = preprocess(sym1) #Preprocess the symptom
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr) #Check similarity with known symptoms
        temp = [sym1, sim1, psym1]
        session['FSY'] = temp  # temp isthe list ,ocntains information about the first symptom (user input,preprocessed symptom , list of all similar diseases)
        session['step'] = "SS"  # next step: Second Symptom
        if sim1 == 1:  # If the first symptom matches a known symptom
            session['step'] = "RS1"  # Transition to Related Symptom step for FS
            s = related_sym(psym1) # Get related symptoms
            if s != 0:
                return s
        else:
            session['step'] = "FS" #if the input from user is gibbrish/does not match any symptom ,then again gaing back to first symp step 
            return "Sorry,I couldn't identify a related symptom for your input. Please try rephrasing or describing another symptom."
    if session['step'] == "RS1":
        temp = session['FSY'] #temp is list ,contains info of first symp 
        psym1 = temp[2] #temp[2] contains the list of possible related symptoms (psym1).
        psym1 = psym1[int(s)] #s is the user's choice, typically an index corresponding to a related symptom presented earlier.The bot uses this index (int(s)) to select the specific related symptom from the list (psym1).
        temp[2] = psym1 #The selected related symptom replaces the list of possible related symptoms in the session data (temp[2]).
        session['FSY'] = temp #The updated information about the first symptom is saved back to the session (session['FSY']).
        session['step'] = 'SS' #step for handling the second symptom ("SS").
        return "Noted!You are probably facing another symptom, if so, can you specify it?"
    if session['step'] == "SS":
        sym2 = s
        sym2 = preprocess(sym2)
        sim2, psym2 = syntactic_similarity(sym2, all_symp_pr)
        temp = [sym2, sim2, psym2]
        session['SSY'] = temp  # information about the Second symptom
        session['step'] = "semantic" 
        if sim2 == 1:
            session['step'] = "RS2"  # related sym2
            s = related_sym(psym2)
            if s != 0:
                return s
        else:  # Invalid symptom input
            session['step'] = 'SS'
            return "I couldn't match your symptom. Please describe it again in simpler terms."
    if session['step'] == "RS2":
        temp = session['SSY']
        psym2 = temp[2]
        psym2 = psym2[int(s)]
        temp[2] = psym2
        session['SSY'] = temp
        session['step'] = "semantic"
    if session['step'] == "semantic":
        temp = session["FSY"]  # Retrieve information from the first sym
        sym1 = temp[0]
        sim1 = temp[1]
        temp = session["SSY"]  # Retrieve information from the 2nd sym 
        sym2 = temp[0]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
        else:
            session['step'] = 'PD'  # to possible_diseases
    if session['step'] == "BFsim1=0":
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symp_pr)
            temp = []
            temp.append(sym1)
            temp.append(sim1)
            temp.append(psym1)
            session['FSY'] = temp
            session['step'] = "sim1=0"  # process of semantic similarity=1 for first sympt.
        else:
            session['step'] = "BFsim2=0"
    if session['step'] == "sim1=0":  # semantic no => suggestion
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure"]:
                    psym1 = sugg[0]
                    sim1 = 1
                    temp = session["FSY"]
                    temp[1] = sim1
                    temp[2] = psym1
                    session["FSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested" not in session:
                session["suggested"] = suggest_syn(sym1)
                sugg = session["suggested"]
            if len(sugg) > 0:
                msg = "are you experiencing any  " + sugg[0] + "?"
                return msg
        if "suggested" in session:
            del session["suggested"]
        session['step'] = "BFsim2=0"
    if session['step'] == "BFsim2=0":
        temp = session["SSY"]  # recuperer info du 2 eme symptome
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symp_pr)
            temp = []
            temp.append(sym2)
            temp.append(sim2)
            temp.append(psym2)
            session['SSY'] = temp
            session['step'] = "sim2=0"
        else:
            session['step'] = "TEST"
    if session['step'] == "sim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure"]:
                    psym2 = sugg[0]
                    sim2 = 1
                    temp = session["SSY"]
                    temp[1] = sim2
                    temp[2] = psym2
                    session["SSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                session["suggested_2"] = suggest_syn(sym2)
                sugg = session["suggested_2"]
            if len(sugg) > 0:
                msg = "Are you experiencing " + sugg[0] + "?"
                session["suggested_2"] = sugg
                return msg
        if "suggested_2" in session:
            del session["suggested_2"]
        session['step'] = "TEST"  # test if semantic and syntaxic and suggestion not found
    if session['step'] == "TEST":
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if sim1 == 0 and sim2 == 0:
            # GO TO THE END
            result = None
            session['step'] = "END"
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["FSY"]
                temp[2] = psym2
                session["FSY"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["SSY"]
                temp[2] = psym1
                session["SSY"] = temp
            session['step'] = 'PD'  # to possible_diseases
    if session['step'] == 'PD':
        # create patient symp list
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if "all" not in session:
            session["asked"] = []
            session["all"] = [col_dict[psym1], col_dict[psym2]]  
        session["diseases"] = possible_diseases(session["all"])
        all_sym = session["all"]
        diseases = session["diseases"]
        session["stored_diseases"] = diseases.copy()

       
        if diseases:
            dis = diseases[0]
            session["dis"] = dis
        else:
            session["dis"] = None
        session['step'] = "for_dis"

    if session['step'] == "DIS":
        if "symv" in session:
            if len(s) > 0 and len(session["symv"]) > 0:
                symts = session["symv"]
                all_sym = session["all"]
                if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure"]:
                    all_sym.append(symts[0])
                    session["all"] = all_sym
                del symts[0]  # Remove the first symptom
                session["symv"] = symts

        if "symv" not in session:
            session["symv"] = symVONdisease(df_tr, session["dis"])

        if len(session["symv"]) > 0:
            symts = session["symv"]
            all_sym = session["all"]
            if symts[0] not in session["all"] and symts[0] not in session.get("asked", []):
                asked = session.get("asked", [])
                asked.append(symts[0])
                session["asked"] = asked
                # msg = "Are you experiencing " + clean_symp(symts[0]) + "?"
                symptom_name = clean_symp(symts[0])  # Clean the symptom name
                description = symptomDescriptionDictionary.get(symptom_name, "Description not available.")
                msg = f"Are you experiencing {symptom_name}? <br>Quick Insight: {description}"
                return msg
            else:
                symptom = symts[0]  # Get the symptom to remove
                session["symv"].remove(symptom)  # Remove symptom from the list
                return get_bot_response()
        else:
            PD = possible_diseases(session["all"])
            diseases = session.get("diseases", [])

            # Remove the disease from the list if all symptoms have been processed
            if len(diseases) > 0:
                processed_disease = diseases[0]
                session["diseases"] = diseases[1:]  # Remove the processed disease

            # If no diseases are left, proceed to the prediction step
            if len(session["diseases"]) == 0:
                session['step'] = 'PREDICT'

            if session["diseases"]:
                session["dis"] = session["diseases"][0]
            else:
                session["dis"] = None

            session['step'] = "for_dis"
            session["symv"] = symVONdisease(df_tr, session["dis"])
            return get_bot_response()

    if session['step'] == "for_dis":
        diseases = session["diseases"]

        if len(diseases) == 0:
            session['step'] = 'PREDICT'
            
        else:
            if session["diseases"]:
                session["dis"] = session["diseases"][0]
            else:
                session["dis"] = None
            session['step'] = "DIS"
            session["symv"] = symVONdisease(df_tr, session["dis"])

            return get_bot_response()

    if session['step'] == "PREDICT":
        result = model.predict(OHV(session["all"], all_symp_col))

        session['step'] = "END"

        if session['step'] == "END":
            if result is not None:
                # Initialize 'stored_diseases' and 'PD' if they do not exist
                if 'stored_diseases' not in session:
                    session['stored_diseases'] = []
                if 'PD' not in session:
                    session['PD'] = []

                # Flow logic:
                if not session['stored_diseases'] and not session['PD']:
                    # If both are empty, assign result[0] to session['disease']
                    session['disease'] = result[0]
                    session['stored_diseases'].append(result[0])
                    session['step'] = "Description"
                    return f"Well, there is a possibility that you may have <span style='color:#2196F3;'> {session['disease']} </span>. This can be confirmed with a proper diagnosis. Type D for Description."


                elif result[0] in session['stored_diseases']:
                    # If result[0] is already in stored_diseases, assign it to session['disease']
                    session['disease'] = result[0]
                    session['step'] = "Description"
                    return f"Well, there is a possibility that you may have <span style='color:#2196F3;'> {session['disease']} </span>. This can be confirmed with a proper diagnosis. Type D for Description."

                else:
                    session['disease']=result[0]
                    session['step'] = "Description"
                    return f"Well, there is a possibility that you may have <span style='color:#2196F3;'> {session['disease']} </span>. This can be confirmed with a proper diagnosis. Type D for Description."

            else:
                # If result is None, ask for more symptoms or offer to end the conversation
                session['step'] = "Q_C"
                return "Can you specify more what you feel or Type q to stop the conversation."

    
    y = {"Name": session["name"], "Age": session["age"], "Gender": session["gender"], "Disease": session["disease"],
              "Symptoms": session["all"]}

    if session['step'] == "Description":
        session['step'] = "Severity"
        
        if s.upper() == "D":
                if session["disease"] in description_list.keys():

                    return description_list[session["disease"]] + " \n <br> Please visit <a href='" + "https://en.wikipedia.org/wiki/" + session["disease"] + "'>here</a> for more information. \n <br>Let me check the severity , How many days have you had symptoms?"
                
                else:
                    if " " in session["disease"]:
                        session["disease"] = session["disease"].replace(" ", "_")
                    return "Please visit <a href='" + "https://en.wikipedia.org/wiki/" + session["disease"] + "'>here</a> for more information. \n <br>Let me check the severity , How many days have you had symptoms?"
        else:
                # Handle other user input if not 'D'
                session['step'] = "Description"
                return "Please type 'D' to get a description of your disease"
        

    if session['step'] == "Severity":
        session['step'] = 'FINAL'
        
        # Use regex to extract the number from the input
        match = re.search(r'\d+', s)  # Find any sequence of digits in the input
        
        if match:
            # Extract the number as an integer
            num_days = int(match.group(0))
            
            # Now proceed with the severity calculation
            if calc_condition(session["all"], num_days) == 1:
                session['severity_flag'] = True
                msg = '<span style="color: red; font-weight: bold;">❗ Seems to be severe! You should take consultation from a doctor asap! ❗</span><br>'
                session['step'] = "ConsultDoctor"  # Transition to next step for precaution
                msg += 'Do you want to consult a doctor? 👩‍⚕️ 🩺 I can help you with that! (yes/no) <br>'
                return msg
            else:
                session['severity_flag'] = False
                msg = '✅ <span style="color: green; font-weight: bold;">Nothing to worry</span>, but you should take the following precautions:<br>'
                i = 1
                for e in precautionDictionary.get(session["disease"], []):  # Safe access of precaution dictionary
                    msg += '\n ' + str(i) + ' - ' + e + '<br>'
                    i += 1
                msg += '🍴 Interested in diet tips? (yes or no)<br> '
                session['step'] = "DietSuggestion"
                return msg
        else:
            # If no number was found, prompt the user to enter a valid number
            session['step'] = "Severity"  
            return "Could you please enter the number of days in numbers? For example, '2 days'."


    if session['step']=="Precautions":
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure","intrested"]:
            msg = f"🍽️ Here are the precautions for {session['disease']}:<br>"
            i = 1
            for e in precautionDictionary.get(session["disease"], []):  # Safe access of precaution dictionary
                msg += '\n ' + str(i) + ' - ' + e + '<br>'
                i += 1
            msg += '🍴 Interested in diet tips? (yes or no)<br> '
            session['step'] = "DietSuggestion"
            return msg
        elif s.lower() in ["no","nope","dont want","don't want","not intrested"]:
            msg = 'Would you like me to provide diet suggestions for this disease? (yes/no) <br>'
            session['step'] = "DietSuggestion"
            return msg
        elif s.lower() in ["q","stop","bye","quit"]:
            session['step'] = "FINAL"
        else:
            msg = 'Please type "yes" to get diet suggestions, "no" to skip and move to medication suggestions <br>'
            return msg


    if session['step'] == "DietSuggestion":
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure","intrested"]:
            msg = f"🍽️ Here are the diet suggestions for {session['disease']}:<br>"
            i = 1

            diets = dietDictionary.get(session["disease"], [])  # Safe access to diet dictionary
            for diet in dietDictionary.get(session["disease"], []):  # get the list of diet suggestions
                msg += '\n ' + str(i) + ' - ' + diet + '<br>'
                i += 1
            msg += ' Do you need medication suggestions for this disease? (yes/no) <br>'
            session['step'] = "MedicationSuggestion"
            return msg
        elif s.lower() in ["no","nope","dont want","don't want","not intrested"]:
            msg = 'Would you like me to provide medication suggestions for this disease? (yes/no) <br>'
            session['step'] = "MedicationSuggestion"
            return msg
        elif s.lower() in ["q","stop","bye","quit"]:
            session['step'] = "FINAL"
        else:
            msg = '⚠️Please type "yes" to get diet suggestions, "no" to skip and move to medication suggestions <br>'
            return msg

        

    if session['step'] == "MedicationSuggestion":
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure","give","show","provide"]:
            msg = f"Here are the medication suggestions for {session['disease']}:<br>"
            i = 1
            medications = medicationDictionary.get(session["disease"], [])  # Safely retrieve medications

            for medication in medicationDictionary.get(session["disease"], []):  # get the list of medication suggestions
                msg += '\n ' + str(i) + ' - ' + medication + '<br>'
                i += 1
            msg += '<span style="color:orange;">⚠️ <strong>Important:</strong> Please consult a doctor before use. ⚠️</span> '

            msg += 'Do you need exercise suggestions for this disease? (yes/no) <br>'
            session['step'] = "ExerciseSuggestion"
            return msg
        elif s.lower() in ["no","nope","dont want","don't want","dont provide"]:
            msg = 'Do you need exercise suggestions for this disease? (yes/no) <br>'
            session['step'] = "ExerciseSuggestion"
            return msg
        elif s.lower() in ["q","stop","bye","quit"]:
            session['step']="Final"
        else:
            msg = "⚠️Could you please answer with either 'Yes' or 'No'? I want to make sure I assist you properly"
            return msg

   
    if session['step'] == "ExerciseSuggestion":
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure","give","show"]:
            if session.get('severity_flag'):
                msg = f"Here are the exercise suggestions for {session['disease']}:<br>"
                i = 1
                exercises = exerciseSuggestionsDictionary.get(session["disease"], [])
                for exercise in exerciseSuggestionsDictionary.get(session["disease"], []):  # get the list of exercise suggestions
                    msg += '\n ' + str(i) + ' - ' + exercise + '<br>'
                    i += 1
                msg += 'do you need any other information? (yes/no)'
                session['step'] = "FINAL"
            else:
                msg = f"Here are the exercise suggestions for {session['disease']}:<br>"
                i = 1
                exercises = exerciseSuggestionsDictionary.get(session["disease"], [])
                for exercise in exerciseSuggestionsDictionary.get(session["disease"], []):  # get the list of exercise suggestions
                    msg += '\n ' + str(i) + ' - ' + exercise + '<br>'
                    i += 1
                msg += 'Do you want to consult a doctor? 👩‍⚕️ 🩺 I can help you with that! (yes/no)'
                session['step'] = "ConsultDoctor"
            return msg

        elif s.lower() in ["no","nope","dont want","don't want"]:
            if session.get('severity_flag'):
                msg = 'do you need any other information? (yes/no)'
                session['step'] = "FINAL"
            else:
                msg = 'Do you want to consult a doctor? 👩‍⚕️ 🩺 I can help you with that! (yes/no)'
                session['step'] = "ConsultDoctor"
            return msg
        elif s.lower() in ["q","stop","bye","quit"]:
            session['step']="Final"
        else:
            msg = "⚠️ Could you please answer with either 'Yes' or 'No'? I want to make sure I assist you properly"
            return msg


    if session['step'] == "ConsultDoctor":
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure"]:
            specialist = specialistDictionary.get(session['disease'], "General Physician") 
            msg = f"For {session['disease']}, it is recommended to consult a {specialist}.👩‍⚕️ <br>"
            msg += 'I can suggest you the hospital and doctor. Do you want me to do so? (yes/no)'
            session['step'] = "HospitalDoctorRecommendation"
            return msg
        elif s.lower() in ["no", "nope", "nah", "not really"]:
            if session.get('severity_flag'):
                msg = "OKay! suggest you some precautions (yes/no)"
                session['step']="Precautions"
            else:
                msg = "📋 Do you need more information or would you like to exit? <br>Type 'q' to end."
                session['step'] = "FINAL"
            return msg

        elif s.lower() in ["q", "quit", "stop", "bye"]:
            session['step'] = "FINAL"

        else:
            # Handle invalid responses
            msg = "⚠️ Please respond with 'yes' or 'no' so I can assist you better."
            return msg

    if session['step'] == "HospitalDoctorRecommendation":
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure","give","show"]:
            # Ask the user for their location (city)
            msg = "Please enter your location (city name):"
            session['step'] = "LocationInput"
            return msg
        elif s.lower() in ["no", "nope", "nah", "not really"]:
            if session.get('severity_flag'):
                msg = "OKay! suggest you some precautions (yes/no)"
                session['step']="Precautions"
            else:
                msg = "📋 Do you need more information or would you like to exit? <br>Type 'q' to end."
                session['step'] = "FINAL"
            return msg
        elif s.lower() in ["q", "quit", "stop", "bye"]:
            session['step'] = "FINAL"
        else:
            msg = 'Do you need any other information? <br> Type q to end.'
            session['step'] = "FINAL"
            return msg
        
    if session['step'] == "LocationInput":
        location = s.strip().lower()  # Normalize input location by stripping spaces and converting to lowercase
        session['location'] = location
        # Get the disease and corresponding specialist
        specialist = specialistDictionary.get(session['disease'], "General Physician")
        msg = f"Here are some recommendations for {specialist} in {location.capitalize()}:<br>"

        # Filter doctors based on location and specialization
        filtered_data = []
        added_doctors = set()  # A set to track added doctors by their name

        for key, doctors in hospitalDoctorDictionary.items():
            # Compare locations after stripping spaces and converting to lowercase
            city_in_dataset = key[1].strip().lower()

            if location == city_in_dataset and key[0] == specialist:  # Match both location and specialization
                for doctor in doctors:
                    # Check if doctor has already been added
                    if doctor['name'] not in added_doctors:
                        # Ensure experience is converted to integer
                        try:
                            doctor['experience'] = int(doctor['experience'])
                        except ValueError:
                            doctor['experience'] = 0  # Handle invalid data gracefully
                        filtered_data.append(doctor)  # Add the doctor to the list
                        added_doctors.add(doctor['name'])  # Mark the doctor as added

        filtered_data = sorted(filtered_data, key=lambda x: x['experience'], reverse=True)
        session['filtered_data'] = filtered_data

        if filtered_data:
            for i, row in enumerate(filtered_data, start=1):
                msg += f"{i} - {row['name']} at {row['hospital']} (Experience: {row['experience']} years)<br>"
            session['filtered_data'] = filtered_data  # Store for further reference
            msg += "Do you need more information about any doctor? (yes/no)"
            session['step'] = "DoctorDetailsQuery"
        else:
            msg += "Sorry, no specific hospital or doctor recommendations available for your location.<br>"
            msg += "Do you want to search in any other location? (yes/no)"
            session['step'] = "AnotherLocationQuery"  # Set step to handle location change

        return msg

    # Handle the response for another location query
    elif session['step'] == "AnotherLocationQuery":
        if s.strip().lower() == "yes":
            msg = "Please provide the location you'd like to search in:"
            session['step'] = "LocationInput"  # Go back to location input
        elif s.strip().lower() == "no":
            msg = "Do you need any other information? (yes/no)"
            session['step'] = "FINAL"  # Move to the final step
        else:
            msg = "I didn't understand that. Do you want to search in another location? Please respond with 'yes' or 'no'."
        return msg


    y["Description"] = description_list[session["disease"]]
    y["Precautions"] = precautionDictionary.get(session["disease"], [])
    y["DietSuggestions"] = dietDictionary.get(session["disease"], [])
    y["MedicationSuggestions"] = medicationDictionary.get(session["disease"], [])
    y["ExerciseSuggestions"] = exerciseSuggestionsDictionary.get(session["disease"], [])
    y["Specialist"] = specialistDictionary.get(session['disease'], "General Physician")
    y["Location"] = session.get('location', "Unknown")
    y["Doctors Recommended"] = session.get('filtered_data', [])

    write_json(y)

    
    if session['step'] == "DoctorDetailsInput":
        filtered_data = session.get('filtered_data', [])
        try:
            # Check if the input is a number (index)
            if s.isdigit():
                index = int(s) - 1
                if 0 <= index < len(filtered_data):
                    doctor = filtered_data[index]
                else:
                    return "⚠️ Invalid number. Please entered valid index or type the full name of the doctor"
            else:
                # Match by doctor's name
                doctor = next((doc for doc in filtered_data if doc['name'].lower() == s.lower()), None)
                if not doctor:
                    return "⚠️ Doctor not found. Please type a valid doctor's name or number."
            
            # Return doctor's details
            msg = f"Details for {doctor['name']}:<br>"
            msg += f"Hospital: {doctor['hospital']}<br>"
            msg += f"Experience: {doctor['experience']} years<br>"
            msg += "Do you need info about any other doctor? (yes/no)"
            session['step'] = "DoctorDetailsQuery"  # Loop back to allow querying another doctor
        except Exception as e:
            msg = "An error occurred. Please try again."
        return msg

    if session['step'] == "DoctorDetailsQuery":
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure"]:
            msg = "Please type the doctor's name or the number beside their name for more details:"
            session['step'] = "DoctorDetailsInput"
        elif s.lower() in ["no","nope","dont want","don't want"]:
            if session.get('severity_flag'):
                msg = "Would you like to know about the precautions(yes/no)"
                session['step'] = "Precautions"
            else:    
                msg = "Do you need any other information?(yes/no)"
                session['step'] = "FINAL"
        else:
            msg = "⚠️ Invalid input. Please respond with 'yes' or 'no'."
        return msg
    

    if session['step'] == "FINAL":
        session['step'] = "BYE"
       
        return "It seems you are done with the diagnosis!📝Do you need another medical consultation (yes or no)? "
    if session['step'] == "BYE":
        name = session["name"]
        age = session["age"]
        gender = session["gender"]
        session.clear()
        if s.lower() in ["yes", "yup", "s", "yeah", "yep", "sure"]:
            session["gender"] = gender
            session["name"] = name
            session["age"] = age
            session['step'] = "FS"
            return "HELLO again, " + session["name"] + " 👋. Please tell me your main symptom."
        elif s.lower() in ["no","nope","dont want","don't want","bye"]:
            return "Thank you for reaching out! 🙏 Wishing you continued good health and well-being.It's always advisable to visit a doctor for regular check-ups and care 🩺👨‍⚕️. Take care, and feel free to reach out again if you need further assistance! I am here whenever you need support! 👋.<br>Click on Print Button to get your diagnosis report as a pdf"   
        else:
            return "Thank you for reaching out! 🙏 Wishing you continued good health and well-being.It's always advisable to visit a doctor for regular check-ups and care 🩺👨‍⚕️. Take care, and feel free to reach out again if you need further assistance! I am here whenever you need support! 👋.<br>Click on Print Button to get your diagnosis report as a pdf"


if __name__ == "__main__":
    import random  # define the random module
    import string

    S = 10  
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
    app.secret_key = str(ran)
    app.run()