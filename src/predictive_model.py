# To-Do: Add Skeleton Code

#knn from sratch

#figure which features of x and y are to these clincal trails. Embedding?

#example to test knn:

import numpy as np


# TODO: Change the distannce to precentage.

#using knn to find a list of best clincal trails for a patient
#if clinal trail and patienrsa have their names in the vector
def knn(patient_to_predict, trials, k):
    best_clinals = []
    
    for trial in trials:
        trial_name = trial[0]
        #get the info of clincal trail and patient
        trial_features = np.array(trial[1:], dtype=float)
        patient_features = np.array(patient_to_predict[1:], dtype=float)

        #get the distance between clianl trail and patient and compute it to probablity
        distance = np.linalg.norm(patient_features - trial_features)
        probability = max(0, 100 - distance * 10)

        #add it to a list
        best_clinals.append((trial_name, probability))

    #sort it to top k best clincal trails
    k_best = sorted(best_clinals, key=lambda x: x[1], reverse=True)[:k]
    
    #format it
    results = []
    for trail_name, prob in k_best:
        results.append(f"{trail_name}, Probablity of success is {prob:2f}%")

    return results

#test examples
new_trials = [
    ['Trial A', 600, 0, 1, 0],
    ['Trial B', 700, 1, 1, 0],
    ['Trial C', 65, 0, 1, 1],
    ['Trial D', 1, 0, 1, 1]
]

new_patient = ['Patient 1', 65, 0, 1, 0]
new_patient2 = ['Patient 2', 10, 0, 1, 0]

new_patients = [new_patient, new_patient2]


#function that lest us find the best clincal trails for each patient in a list of patients
def knn_mutiple_patients(patients, trials, k):
    patients_too_match_with_ct = []
    #go through the list of pateinst and perform knn of them to the clinal trails
    for unique_patient in patients:
        patient_name = unique_patient[0]
        matches = knn(unique_patient, trials, k)
        patients_too_match_with_ct.append(f"{patient_name} : {matches}")
    return patients_too_match_with_ct


test = knn_mutiple_patients(new_patients, new_trials, 2)
print(test)