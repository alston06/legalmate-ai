import os
import glob
import zipfile
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from io import BytesIO
import re

# --- Configuration ---
# Path to the directory containing the 'year=XXXX' folders (relative to this script)
BASE_DATA_PATH = 'ai-models/data/supreme_court' 
FAISS_INDEX_PATH = 'embeddings/sc_judgments_faiss_index.bin'
METADATA_PATH = 'embeddings/sc_judgments_metadata.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2' 
# ---------------------

def simple_chunking(text, min_length=100):
    """Splits text by double newlines and filters out short chunks."""
    # Split the text into paragraphs (chunks)
    paragraphs = text.split('\n\n')
    chunks = []
    for chunk in paragraphs:
        chunk = chunk.strip()
        if len(chunk) > min_length:
            chunks.append(chunk)
    return chunks

def extract_and_chunk_sc_judgments(base_path, all_chunks):
    """
    Finds all ZIP files, extracts PDFs in memory, and converts their content into chunks.
    """
    # Find all ZIP files recursively (using the ** to search nested year folders)
    zip_files = [
        f for f in glob.glob(os.path.join(base_path, '**', '*.zip'), recursive=True) 
        if 'english' in os.path.basename(f)
    ]
    
    if not zip_files:
        print("❌ No ZIP files found in the Supreme Court data path.")
        return all_chunks

    print(f"Found {len(zip_files)} ZIP archives to process.")

    for zip_path in zip_files:
        print(f"Processing ZIP: {zip_path}")
        
        try:
            # Extract Year from the directory structure (e.g., year=1984)
            year_match = next((part.split('=')[1] for part in zip_path.split(os.sep) if part.startswith('year=')), 'Unknown')
            
            with zipfile.ZipFile(zip_path, 'r') as archive:
                # Iterate through all PDFs inside the ZIP
                pdf_files = [name for name in archive.namelist() if name.lower().endswith('.pdf')]
                
                for pdf_name in pdf_files:
                    try:
                        # Extract PDF content in memory (BytesIO)
                        with archive.open(pdf_name) as pdf_file:
                            reader = PdfReader(BytesIO(pdf_file.read()))
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() + "\n\n"
                        
                        case_id = pdf_name.replace('.pdf', '')
                        
                        # Apply chunking strategy
                        content_chunks = simple_chunking(text)
                        
                        # Package the chunks and metadata
                        for i, chunk in enumerate(content_chunks):
                            metadata = {
                                'source_zip': os.path.basename(zip_path),
                                'case_id': case_id,
                                'year': year_match,
                                'legal_domain': 'Supreme Court Judgment',
                                'chunk_index': i
                            }
                            # Prepend key metadata to the text for better embedding similarity
                            full_chunk_text = f"Case: {case_id}, Year: {year_match}.\nContent: {chunk}"
                            all_chunks.append({'text': full_chunk_text, 'metadata': metadata})

                    except Exception as e:
                        print(f"  Warning: Skipping PDF {pdf_name} due to error: {e}")
                        continue

        except Exception as e:
            print(f"Error processing ZIP {zip_path}: {e}")
            continue

    return all_chunks

# ----------------------------------------------------
# --- Main Execution Block ---
# ----------------------------------------------------
if __name__ == "__main__":
    
    all_chunks = []
    
    # 1. Load the Embedding Model
    print("Loading Sentence Transformer model. This may take a moment...")
    try:
        embed_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"FATAL: Could not load embedding model. Ensure internet connection or check installation. Error: {e}")
        exit()

    # 2. Extract and Chunk SC Judgments
    all_chunks = extract_and_chunk_sc_judgments(BASE_DATA_PATH, all_chunks)
    
    if not all_chunks:
        print("\nNo chunks created. Aborting indexing.")
        exit()

    print(f"\nTotal chunks created: {len(all_chunks)}")
    
    # 3. Create Embeddings
    print("Generating embeddings for all chunks...")
    texts_to_embed = [chunk['text'] for chunk in all_chunks]
    embeddings = embed_model.encode(texts_to_embed, convert_to_tensor=False)
    D = embeddings.shape[1] 

    # 4. Create and Save the FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(D) 
    index.add(embeddings)
    
    faiss.write_index(index, FAISS_INDEX_PATH)

    # 5. Save Metadata
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)

    print("\n--- Indexing Complete ---")
    print(f"✅ FAISS Index saved to {FAISS_INDEX_PATH}")
    print(f"✅ Metadata saved to {METADATA_PATH}")
    print(f"Index size: {index.ntotal} documents.")