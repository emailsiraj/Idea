from google_Search import policy_pdf_search
import os
from pathlib import Path

# Cross-platform file path
file_path = Path(__file__).parent / "poli_nm_parser" / "Collin_plans.txt"

# Check if file already exists
if file_path.exists():
    raise FileExistsError("file collin_plans.txt already exist please delete the old file first")

with open(file_path) as f:
    content = f.read()

plcy_nm = [i for i in content.split('\n')]

# srch_plcy = plcy_nm[0:5]

for plcy in srch_plcy:
    policy_pdf_search('Collin', plcy)