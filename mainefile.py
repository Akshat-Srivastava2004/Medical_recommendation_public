# %%
welcome = "welcome to the medical recommendation system"
print(welcome)

# %%
import pandas as pd 
df = pd.read_csv('Training.csv')

# %%
df.head()

# %%
df.shape

# %%
df['prognosis'].unique()

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
X=df.drop("prognosis",axis=1)
Y=df['prognosis']

# %%
le=LabelEncoder()
le.fit(Y)
Y_Transform=le.transform(Y)

# %%
Y_Transform

# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,Y_Transform,test_size=0.3,random_state=20)

# %%
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np


# %%
#Create a dictionary to store model
models={
    "SVC":SVC(kernel='linear'),
   #  "RandomForestClassifier":RandomForestClassifier(n_estimators=100,random_state=42),
   #  "GradientBoostingClassifier":GradientBoostingClassifier(n_estimators=100,random_state=42),
   #  "KNeighborsClassifier":KNeighborsClassifier(n_neighbors=5),
   #  "MultinomialNB":MultinomialNB()
}
for model_name,model in models.items():
    #train model
    model.fit(X_train,Y_train)
    #test model
    predictions = model.predict(X_test)

    # calculate accuracy
    accuracy=accuracy_score(Y_test,predictions)

    #calculate confusion matrix 
    cm=confusion_matrix(Y_test,predictions)

   #  print(f"{model_name}accuracy:{accuracy}")
   #  print(f"{model_name}confusion Matrix:")
   #  print(np.array2string(cm,separator=','))


# %%
svc = SVC(kernel='linear')
svc.fit(X_train,Y_train)
ypred=svc.predict(X_test)
accuracy_score(Y_test,ypred)

# %%
#saving model
import pickle
pickle.dump(svc,open("models/svc.pkl",'wb'))

# %%
#load model
svc=pickle.load(open("models/svc.pkl",'rb'))

# %%
#test 1
print("model predicted disease :",svc.predict(X_test.iloc[0].values.reshape(1,-1)))
print("Actual label:",Y_test[0])


# %%
#test 2 
print("model predicted disease :",svc.predict(X_test.iloc[10].values.reshape(1,-1)))
print("Actual label:",Y_test[10])


# %%
precaution=pd.read_csv('precautions_df.csv')
workout=pd.read_csv('workout_df.csv')
description=pd.read_csv('description.csv')
medication=pd.read_csv('medications.csv')
diets=pd.read_csv('diets.csv')

# %%
#helper function

def helper(dis):
    # Get description
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])  # Join description into a single string
    
    # Get precautions
    pre = precaution[precaution['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    pre = " | ".join([str(item) for sublist in pre for item in sublist if item])  # Convert list of lists into a string

    # Get medications
    med = medication[medication['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    med = " | ".join([str(m) for m in med if m])  # Join medications into a single string

    # Get diets
    die = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die.values]
    die = " | ".join([str(d) for d in die if d])  # Join diets into a single string

    # Get workouts
    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [w for w in wrkout.values]
    wrkout = " | ".join([str(w) for w in wrkout if w])  # Join workouts into a single string

    # Print types of each variable
    print("Type of description:", type(desc))
    print("Type of precautions:", type(pre))
    print("Type of medications:", type(med))
    print("Type of diets:", type(die))
    print("Type of workouts:", type(wrkout))

    return desc, pre, med, die, wrkout



#model predictions function
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_predicted_value(patient_symptoms):
    print("THE VALUE IS COMING", patient_symptoms)
    input_vector = np.zeros(len(symptoms_dict))
    
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    
    predicted_disease = diseases_list[svc.predict([input_vector])[0]]
    
    # Check and print the type of the predicted disease
    print("Type of predicted disease:", type(predicted_disease))
    
    return predicted_disease

# %%
# symptoms=input("Enter your symptoms...........")
# user_symptoms=[s.strip() for s in symptoms.split(',')]
# user_symptoms=[sym.strip("[]' ") for sym in user_symptoms]
# predicted_disease=get_predicted_value()
# desc,pre,med,die,wrkout=helper(predicted_disease)



# print("======================================Predicted Disease================================")

# print(predicted_disease)

# print("=======================================Description============================")

# print(desc)

# print("=======================================Precaution============================")
# for p_i in pre[0]:
#    j=1
#    print(j,":",p_i)
#    j+=1

# print("=======================================medicination============================")
# for m_i in med:
#    j=1
#    print(j,":",m_i)
#    j+=1
#    print("=======================================Workout=====================")
   
# for w_i in wrkout:
#    j=1
#    print(j,":",w_i)
#    j +=1
# print("=======================================Diet=================")
# for d in die:
#    j=1
#    print(j,":",d)
#    j +=1

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd  # Assuming pandas is also being used
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Medical Recommendation System!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receive JSON input
    symptoms = data.get('symptoms', [])  # Get symptoms from request
    
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    
    # Ensure `symptoms` is converted to a list
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]

    # Predict disease
    predicted_disease = get_predicted_value(user_symptoms)
    print("The disease is:", predicted_disease)
    
    # Get additional details
    desc, pre, med, die, wrkout = helper(predicted_disease)
    print("Description:", desc)
    print("Precautions:", pre)
    print("Medications:", med)
    print("Workouts:", wrkout)

    # Convert pandas Series or numpy ndarray to list if needed
    def convert_to_list(data):
        if isinstance(data, pd.Series):
            return data.tolist()
        elif isinstance(data, np.ndarray):  # Handling ndarray
            return data.tolist()
        elif isinstance(data, list):  # If already a list, return as is
            return data
        else:
            return [data]  # Wrap non-list data into a list (if required)

    # Convert all lists or non-serializable types to proper lists
    desc = convert_to_list(desc)
    pre = convert_to_list(pre)
    med = convert_to_list(med)
    wrkout = convert_to_list(wrkout)

    # Prepare the response data
    response = {
       "disease": predicted_disease,
        "description": desc,
        "precautions": pre,
        "medications": med,
        "diets": die,
        "workouts": wrkout 
    }

    # Return JSON response
    return jsonify(response)


if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5000)




# %%



