import pandas as pd
import re
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader # Import the PDF reader library

# --- Configuration ---
PDF_PATH = 'ai-models/data/ConstitutionOfIndia.pdf' # Path relative to the script location
FAISS_INDEX_PATH = 'embeddings/constitution_faiss_index.bin'
METADATA_PATH = 'embeddings/constitution_metadata.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2' 
# ---------------------

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred during PDF extraction: {e}")
        return None

def chunk_legal_text(text, source_name="Unknown Act", legal_domain="General Law"):
    """
    Chunks text based on the pattern 'Article X:' (or a similar pattern for an Act) 
    and adds metadata.
    """
    # NOTE: You will need to adjust this regex based on the exact format of your PDF.
    # We'll use a general "ARTICLE X" pattern for now.
    chunks = re.split(r'(ARTICLE\s\d+\.?(\([A-Za-z]\))?(\s\(.{1,5}\))?:?|\n[0-9]{1,3}\.\s)', text)
    
    data = []
    current_title = ""
    
    for part in chunks:
        if part and len(part.strip()) > 5:
            # Check if the part looks like a new section/article title
            is_title = re.match(r'(ARTICLE\s\d+)', part.strip())
            
            if is_title:
                current_title = part.strip().split('\n')[0] # Use only the first line as title
            
            # Combine the section title (if found) with the content
            full_text = f"Source: {source_name}\nSection/Article: {current_title or 'Start'}\nContent: {part.strip()}"

            metadata = {
                'source': source_name,
                'section_title': current_title or 'Introduction',
                'legal_domain': legal_domain
            }
            data.append({'text': full_text, 'metadata': metadata})
    
    # A cleaner chunking approach for Statutes/Acts often involves splitting by section headers
    # and ensuring the full content of the section goes with its header.
    # This simple method may require refinement based on your specific PDF's formatting.
    return data


# --- Main Execution ---

# 1. Extract Text
print("Starting text extraction from PDF...")
raw_text = extract_text_from_pdf(PDF_PATH)

if raw_text is None:
    exit()

# 2. Chunk Text
# Adjust 'source_name' and 'legal_domain' as needed
legal_chunks = chunk_legal_text(raw_text, "Constitution of India, 1950", "Constitutional Law")
print(f"Successfully split document into {len(legal_chunks)} chunks.")

# 3. Create Embeddings
print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
texts_to_embed = [chunk['text'] for chunk in legal_chunks]
embeddings = model.encode(texts_to_embed, convert_to_tensor=False)
D = embeddings.shape[1] 

# 4. Create and Save the FAISS Index
index = faiss.IndexFlatL2(D) 
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)

# 5. Save Metadata
with open(METADATA_PATH, 'wb') as f:
    pickle.dump(legal_chunks, f)

print("\n--- Process Complete ---")
print(f"✅ FAISS Index saved to {FAISS_INDEX_PATH}")
print(f"✅ Metadata saved to {METADATA_PATH}")
print(f"Total indexed chunks: {index.ntotal}")