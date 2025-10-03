import os
import glob
import zipfile
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from io import BytesIO
import time 
import re

# --- Configuration ---
BASE_DATA_PATH = 'ai-models/data/supreme_court' 
FAISS_INDEX_PATH = 'embeddings/sc_judgments_faiss_index.bin'
METADATA_PATH = 'embeddings/sc_judgments_metadata.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2' 
# ---------------------

def simple_chunking(text, min_length=100):
    """Splits text by double newlines and filters out short chunks."""
    paragraphs = text.split('\n\n')
    chunks = []
    for chunk in paragraphs:
        chunk = chunk.strip()
        if len(chunk) > min_length:
            chunks.append(chunk)
    return chunks

def load_or_create_index(D):
    """Loads existing FAISS index and metadata or creates new ones."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            print("Loading existing index and metadata for incremental update...")
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
            return index, metadata
        except Exception as e:
            print(f"Error loading existing index. Starting fresh. Error: {e}")
            return faiss.IndexFlatL2(D), []
    else:
        print("No existing index found. Starting fresh...")
        return faiss.IndexFlatL2(D), []

def save_index_and_metadata(index, metadata):
    """Saves the FAISS index and metadata safely."""
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ SAVE CHECKPOINT: Index saved with {index.ntotal} total documents.")
    except Exception as e:
        print(f"❌ FATAL SAVE ERROR: Could not save index/metadata: {e}")


def process_zip_incrementally(zip_path, embed_model, index, all_metadata):
    """Processes a single ZIP, generates embeddings, and adds to the master index."""
    
    zip_basename = os.path.basename(zip_path)
    new_chunks = []

    try:
        year_match = next((part.split('=')[1] for part in zip_path.split(os.sep) if part.startswith('year=')), 'Unknown')
        
        with zipfile.ZipFile(zip_path, 'r') as archive:
            pdf_files = [name for name in archive.namelist() if name.lower().endswith('.pdf')]
            
            for pdf_name in pdf_files:
                # ... (PDF reading and chunking logic remains the same) ...
                safe_pdf_name = pdf_name.replace('\\', '/') 
                case_id = pdf_name.replace('.pdf', '')
                
                try:
                    with archive.open(safe_pdf_name) as pdf_file:
                        reader = PdfReader(BytesIO(pdf_file.read()))
                        text = "".join([page.extract_text() + "\n\n" for page in reader.pages])
                    
                    content_chunks = simple_chunking(text)
                    
                    if len(content_chunks) == 0:
                        continue
                        
                    for i, chunk in enumerate(content_chunks):
                        metadata = {
                            'source_zip': zip_basename,
                            'case_id': case_id,
                            'year': year_match,
                            'legal_domain': 'Supreme Court Judgment',
                            'chunk_index': i
                        }
                        full_chunk_text = f"Case: {case_id}, Year: {year_match}.\nContent: {chunk}"
                        new_chunks.append({'text': full_chunk_text, 'metadata': metadata})

                except Exception as e:
                    print(f"  !! CRITICAL ERROR processing PDF {safe_pdf_name} in {zip_basename}: {e}")
                    continue
        
        # --- Core Incremental Logic ---
        if new_chunks:
            # 1. Generate embeddings for ONLY the new chunks
            texts_to_embed = [chunk['text'] for chunk in new_chunks]
            embeddings = embed_model.encode(texts_to_embed, convert_to_tensor=False)
            
            # 2. Add embeddings to the master index
            index.add(embeddings)
            
            # 3. Add metadata to the master list
            all_metadata.extend(new_chunks)
            
            print(f"  -> SUCCESS: Indexed {len(new_chunks)} new chunks from {zip_basename}.")
            
    except Exception as e:
        print(f"Error opening ZIP {zip_path}: {e}")
        
    return index, all_metadata

# ----------------------------------------------------
# --- Main Execution Block ---
# ----------------------------------------------------
if __name__ == "__main__":
    
    start_time = time.time()
    
    # 1. Load the Embedding Model
    print("Loading Sentence Transformer model...")
    embed_model = SentenceTransformer(MODEL_NAME)
    D = embed_model.get_sentence_embedding_dimension()

    # 2. Load existing or create new master index
    master_index, master_metadata = load_or_create_index(D)
    
    # 3. Find and Filter English ZIPs
    all_zip_files = glob.glob(os.path.join(BASE_DATA_PATH, '**', '*.zip'), recursive=True)
    zip_files = [f for f in all_zip_files if 'english' in os.path.basename(f).lower()]
    
    if not zip_files:
        print("\nIndexing failed. No English ZIP files found.")
        exit()

    # 4. Iterate, Process, and Save Incrementally
    initial_count = master_index.ntotal
    
    for i, zip_path in enumerate(zip_files):
        print(f"\n--- BATCH {i+1}/{len(zip_files)}: {os.path.basename(zip_path)} ---")
        
        # This function processes the ZIP and updates the master index/metadata lists
        master_index, master_metadata = process_zip_incrementally(zip_path, embed_model, master_index, master_metadata)
        
        # Save after every ZIP file
        save_index_and_metadata(master_index, master_metadata)
        
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    print("\n=====================================")
    print("INCREMENTAL INDEXING COMPLETE!")
    print(f"Total Chunks Indexed: {master_index.ntotal}")
    print(f"Chunks Added in this Run: {master_index.ntotal - initial_count}")
    print(f"Processing Time: {total_time:.2f} minutes")
    print("=====================================")