import os
from pathlib import Path
import fitz
import re

#Get pdfs

#Testing with local folder with "./uploads/"
single_PDF = "...pdf"

#For backend team to upload multiple protocols
UPLOAD_DIR = Path("./uploads")

PROTOCOL_PATTERN = re.compile(r'^Protocol\s+(A-Z){2}\d{5}', re.IGNORECASE)

def get_protocol_pdfs(upload_dir):

    pdfs = []

    for file in os.listdir(upload_dir):
        if file.lower().startswith("protocol") and file.lower().endswith(".pdf"):
            pdfs.append(upload_dir / file)
    return pdfs

def extract_text_from_pdf(upload_dir):

    extracted_data = []

    for pdf_path in get_protocol_pdfs(upload_dir):
        filename = pdf_path.name
        match = PROTOCOL_PATTERN.match(filename)
        if not match:
            print(f"Could not extract protocol number from: {filename}")
            continue

    protocol_number = match.group(1)

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    extracted_data.append({
       "protocol_number": protocol_number,
        "source_file": filename,
        "text": full_text
    })

    test = fitz.open(single_PDF) #take whole sction out after testing
    single_text = ""
    for page in doc:
        single_text += page.get_text()
    doc.close()

    single_extraction = ({
        "protocol_number": protocol_number,
        "source_file": filename,
        "text": single_text
    })

    return extracted_data, single_extraction #take it out after testing






# expected file format
# {
#   "protocol_number": "AB12345"
#   "source_file": "Protocol AB12345 ...."
#   "text": "..........................."
