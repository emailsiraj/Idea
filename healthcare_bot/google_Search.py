from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
import json
from pathlib import Path

# Load environment variables
load_dotenv()
# print(os.getenv('MYVAR'))
OUTPUT_FOLDER = Path.home() / "OneDrive" / "Desktop" / "SIraj" / "policy_json"

def policy_pdf_search(county,plcy_name):
     # Function to perform policy PDF search
 
    params = {
        "api_key": os.getenv('MYVAR'),
        "engine": "google",
        'q': f'"2026 {plcy_name}" ("Summary of Benefits") filetype:pdf',
        "location": "Austin, Texas, United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # Build the output file path inside the folder
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / f"{county}_{plcy_name}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Response saved to: {output_file}")
