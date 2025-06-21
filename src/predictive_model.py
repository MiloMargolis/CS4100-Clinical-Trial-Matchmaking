"""
This file is designed for KNN predictions to best match patients to clinical trials.
"""

import numpy as np

# using knn to find a list of the best clinical trails for a patient
# if clinal trail and patients have their names in the vector

def knn(patient_to_predict, trials, k):
    # get all distances between the patient and each trial
    all_distances = []
    for trial in trials:
        trial_vector = np.array(trial[1:], dtype=float)
        patient_vector = np.array(patient_to_predict[1:], dtype=float)
        all_distances.append(np.linalg.norm(patient_vector - trial_vector))

    scored = []
    for (trial, distance) in zip(trials, all_distances):
        name = trial[0] # current name of this trail
        # matching score formula:
        score = 100 / (1 + distance)
        scored.append((name, score)) 
    
    scored.sort(key=lambda item: item[1], reverse=True)
    topk = scored[:k]

    return ["{0}, Match Score = {1:.2f}%".format(name, score) for name, score in topk]
    

# old code from alex 

#test examples
new_trials = [
    ['Trial A', 600, 0, 1, 0],
    ['Trial B', 700, 1, 1, 0],
    ['Trial C', 65, 0, 1, 1],
    ['Trial D', 1, 0, 1, 1]
]

new_patient1 = ['Patient 1', 65, 0, 1, 0]
new_patient2 = ['Patient 2', 10, 0, 1, 0]

new_patients = [new_patient1, new_patient2]

#function that lest us find the best clinical trails for each patient in a list of patients
def knn_multiple_patients(patients, trials, k):
    patients_too_match_with_ct = []
    # go through the list of patients and perform knn of them to the clinal trails
    for unique_patient in patients:
        patient_name = unique_patient[0]
        matches = knn(unique_patient, trials, k)
        patients_too_match_with_ct.append(f"{patient_name} : {matches}")
    return patients_too_match_with_ct


test = knn_multiple_patients(new_patients, new_trials, 2)
print(test)
