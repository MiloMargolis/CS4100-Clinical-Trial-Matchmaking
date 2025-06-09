# To-Do: Add Skeleton Code

#knn from sratch

#figure which features of x and y are to these clincal trails. Embedding?

#example to test knn:

import numpy as np

clinical_trials = np.array([
    [60, 0, 1, 0],   
    [70, 1, 1, 0],   
    [65, 0, 0, 1],   
    [64, 0, 1, 0],   
    [66, 1, 1, 0],  
])

trial_names = ['Trial A', 'Trial B', 'Trial C', 'Trial D', 'Trial E']


patient = np.array([65, 0, 1, 0])


# TODo Will change the distannce to precentage?

def knn(patent_to_predict, trails, trial_names, k):
    distances_list = []
    for trail in trails:
        #getting the distance of the new point to every trail.
        distance = np.linalg.norm(patent_to_predict - trail)


        #if clinal trails is one varibale:
        #distance = np.linalg.norm(patent_to_predict - trails[features])

        #putting it in a list
        distances_list.append(distance)

    #sort the distances to get the k closest to the new point/pateint  
    nearest_clinical = np.argsort(distances_list)[:k]
    
    #get the trail names, distances/(percentage later), and k top trails
    best_trials = [(trial_names[i], distances_list[i]) for i in nearest_clinical]

    return best_trials
    
    

result = knn(patient, clinical_trials, trial_names, 5)

print(result)
