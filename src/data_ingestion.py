import requests
import pandas as pd

"""
This script fetches clinical trial data from the ClinicalTrials.gov v2 API
using the GET /studies endpoint. It uses a search filter (currently using
'query.cond') to find trials by condition (default: "lung cancer"). The script
extracts key fields (NCTId, Title, Condition, Status) from the JSON response,
flattens them into a pandas DataFrame, prints the first few rows, and saves the
data to 'data/clinical_trials.csv' for later use.

Usage:
- Run this script from the project root directory with:
    python3 src/data_ingestion.py
- The script will print the first few rows and save the CSV.

Future Improvements:
- Add more fields (Eligibility Criteria, Phase, etc.)
- Implement pagination for more than 100 trials.
- Error handling and logging.
"""


def fetch_trials_v2(condition="lung cancer", page_size=100):
    base_url = "https://beta-ut.clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,  # e.g. "lung cancer"
        "pageSize": page_size
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    data = response.json()
    studies = data.get("studies", [])

    # Flatten data into DataFrame (example with id, title, status)
    records = []
    for study in studies:
        record = {
            "NCTId": study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", ""),
            "Title": study.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
            "Condition": ", ".join(study.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", [])),
            "Status": study.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", "")
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = fetch_trials_v2()
    print(df.head())
    df.to_csv("data/clinical_trials.csv", index=False)
    print("Saved clinical_trials.csv successfully.")
