import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

clinical_trials = np.array([
    [40, 0, 1, 0],   
    [75, 1, 1, 0],   
    [65, 0, 0, 1],   
    [24, 0, 1, 0],   
    [12, 0, 1, 0],  
])
trial_names = ['Trial A', 'Trial B', 'Trial C', 'Trial D', 'Trial E']
patient = np.array([65, 0, 1, 0])


# TODO: Change the distannce to precentage.
# And Change so that knn handles if trails would include their name/lbale and number/info in the same variable

#using knn to find a list of best clincal trails for a patient
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
    
#when we want to find the best clincal trails for a list of patients    
def knn_mutiple_patients(patients, trails, trial_names, k):
    patients_with_ct = []
    for unique_patient in patients:
        matches = knn(unique_patient, trails, trial_names, k)
        patients_with_ct.append(matches)
    return patients_with_ct


result = knn(patient, clinical_trials, trial_names, 5)

print(result)

# new 

# 
# decision tree code
# 
def train_decision_tree(X: np.ndarray, y: np.ndarray,
    *, max_depth: int | None = None, criterion: str = "gini",
    test_size: float = 0.2, random_state: int = 42,) -> DecisionTreeClassifier:

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)

    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print("Decision Tree's Accuracy: {:.3f}".format(acc))
    return clf


def predict_decision_tree(clf: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
    # Predict with a trained decision tree.
    return clf.predict(X)


# 
# visualisation for decision tree
# 
def visualize_decision_tree(clf: DecisionTreeClassifier, *, feature_names: list[str] | None = None,
    class_names: list[str] | None = None, figsize: tuple[int, int] = (18, 8), save_path: str | None = None,):

    plt.figure(figsize=figsize)
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10,)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    patient_csv = DATA_DIR / "patient_embeddings.csv"
    trial_csv = DATA_DIR / "trial_embeddings.csv"

    # -----------------------------------------------------------------------
    # gpt csv file demo
    # -----------------------------------------------------------------------

    # patient_df = pd.read_csv("/Users/neyonought/Documents/CS4100-Clinical-Trial-Matchmaking/data/patient_embeddings.csv")
    # trial_df = pd.read_csv("/Users/neyonought/Documents/CS4100-Clinical-Trial-Matchmaking/data/trial_embeddings.csv")
    # patient_csv = Path("/Users/neyonought/Documents/CS4100-Clinical-Trial-Matchmaking/data/patient_embeddings.csv")
    # trial_csv = Path("/Users/neyonought/Documents/CS4100-Clinical-Trial-Matchmaking/data/trial_embeddings.csv")

    if patient_csv.exists() and trial_csv.exists():
        patient_df = pd.read_csv(patient_csv)
        trial_df = pd.read_csv(trial_csv)

        patient_embeds = patient_df.iloc[:, 1:].values  # skip ID column
        trial_embeds = trial_df.iloc[:, 1:].values
        trial_ids = trial_df.iloc[:, 0].astype(str).tolist()

        n_pairs = min(len(patient_embeds), len(trial_embeds))
        X_real = np.abs(patient_embeds[:n_pairs] - trial_embeds[:n_pairs])

        y_real = np.random.randint(0, 2, size=n_pairs)

        tree_real = train_decision_tree(X_real, y_real, max_depth=4)
        visualize_decision_tree(
            tree_real,
            feature_names=[f"f{i}" for i in range(X_real.shape[1])],
            class_names=["No‑Match", "Match"],
        )
    else:
        print("[skip] CSV files not found – set CTM_DATA_DIR or place CSVs in ./data/")