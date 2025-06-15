import requests
import pandas as pd

"""
This script fetches up to 500 clinical trials from the ClinicalTrials.gov v2 API
based on a condition (default: "lung cancer"). Results are saved to
'data/clinical_trials.csv'.
"""

def fetch_trials_v2(condition="lung cancer", total_desired=500, page_size=100):
    base_url = "https://beta.clinicaltrials.gov/api/v2/studies"
    all_records = []
    next_page_token = None

    while len(all_records) < total_desired:
        params = {
            "query.cond": condition,
            "pageSize": page_size,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            idmod = study.get("protocolSection", {}).get("identificationModule", {})
            condmod = study.get("protocolSection", {}).get("conditionsModule", {})
            statmod = study.get("protocolSection", {}).get("statusModule", {})
            eligmod = study.get("protocolSection", {}).get("eligibilityModule", {})
            designmod = study.get("protocolSection", {}).get("designModule", {})

            record = {
                "NCTId": idmod.get("nctId", ""),
                "Title": idmod.get("briefTitle", ""),
                "Condition": ", ".join(condmod.get("conditions", [])),
                "Status": statmod.get("overallStatus", ""),
                "Eligibility": eligmod.get("eligibilityCriteria", ""),
                "Phase": ", ".join(designmod.get("phaseList", {}).get("phases", [])),
                "Enrollment": designmod.get("enrollmentInfo", {}).get("enrollmentCount", ""),
                "StudyType": designmod.get("studyType", "")
            }

            all_records.append(record)

        next_page_token = data.get("nextPageToken", None)
        if not next_page_token:
            break

    return pd.DataFrame(all_records[:total_desired])


if __name__ == "__main__":
    df = fetch_trials_v2(total_desired=500)
    print(f"Fetched {len(df)} studies")
    print(df["Eligibility"].head(10))
    df.to_csv("data/clinical_trials.csv", index=False)
    print("Saved clinical_trials.csv successfully.")
