<<<<<<< HEAD
import pandas as pd
from sklearn.utils import shuffle

def extract_symptoms_from_combined_data(data_path, state):
    train_data = pd.read_csv(data_path)

    complaints = []
    symptoms = []

    for indexi in range(0, len(train_data["symptoms"])):
        complaint = train_data["symptoms"][indexi].split("[SEP]")[0]
        symptom = train_data["symptoms"][indexi].split("[SEP]")[1].lstrip()

        complaints.append(complaint)
        symptoms.append(symptom)

    all_complaints = pd.DataFrame()
    all_symptoms = pd.DataFrame()


    dataset = pd.DataFrame()
    dataset["diagnosis"] = train_data["diagnosis"]
    dataset["complaints"] = complaints
    dataset["symptoms"] = symptoms

    dataset = shuffle(dataset)

    all_complaints["diagnosis"] = dataset["diagnosis"]
    all_complaints["symptoms"] = dataset["complaints"]

    all_symptoms["diagnosis"] = dataset["diagnosis"]
    all_symptoms["symptoms"] = dataset["symptoms"]


    # all_complaints["diagnosis"] = train_data["diagnosis"]
    # all_complaints["symptoms"] = complaints
    #
    # all_symptoms["diagnosis"] = train_data["diagnosis"]
    # all_symptoms["symptoms"] = symptoms

    all_complaints.to_csv("./data/complaint_" + state + ".csv", index=False)
    all_symptoms.to_csv("./data/symptom_" + state + ".csv", index=False)



train_data_path = "../data/processed_training_data/combine_complaint_symp/train.csv"
test_data_path = "../data/processed_training_data/combine_complaint_symp/train.csv"
val_data_path = "../data/processed_training_data/combine_complaint_symp/train.csv"


extract_symptoms_from_combined_data(train_data_path, "train")
extract_symptoms_from_combined_data(test_data_path, "test")
=======
import pandas as pd
from sklearn.utils import shuffle

def extract_symptoms_from_combined_data(data_path, state):
    train_data = pd.read_csv(data_path)

    complaints = []
    symptoms = []

    for indexi in range(0, len(train_data["symptoms"])):
        complaint = train_data["symptoms"][indexi].split("[SEP]")[0]
        symptom = train_data["symptoms"][indexi].split("[SEP]")[1].lstrip()

        complaints.append(complaint)
        symptoms.append(symptom)

    all_complaints = pd.DataFrame()
    all_symptoms = pd.DataFrame()


    dataset = pd.DataFrame()
    dataset["diagnosis"] = train_data["diagnosis"]
    dataset["complaints"] = complaints
    dataset["symptoms"] = symptoms

    dataset = shuffle(dataset)

    all_complaints["diagnosis"] = dataset["diagnosis"]
    all_complaints["symptoms"] = dataset["complaints"]

    all_symptoms["diagnosis"] = dataset["diagnosis"]
    all_symptoms["symptoms"] = dataset["symptoms"]


    # all_complaints["diagnosis"] = train_data["diagnosis"]
    # all_complaints["symptoms"] = complaints
    #
    # all_symptoms["diagnosis"] = train_data["diagnosis"]
    # all_symptoms["symptoms"] = symptoms

    all_complaints.to_csv("./data/complaint_" + state + ".csv", index=False)
    all_symptoms.to_csv("./data/symptom_" + state + ".csv", index=False)



train_data_path = "../data/processed_training_data/combine_complaint_symp/train.csv"
test_data_path = "../data/processed_training_data/combine_complaint_symp/train.csv"
val_data_path = "../data/processed_training_data/combine_complaint_symp/train.csv"


extract_symptoms_from_combined_data(train_data_path, "train")
extract_symptoms_from_combined_data(test_data_path, "test")
>>>>>>> a39879c (first upload)
extract_symptoms_from_combined_data(val_data_path, "val")