import torch
import pandas as pd
import faiss
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class FinancialChatbot:
    def __init__(self, data_path, model_name="all-MiniLM-L6-v2", qwen_model_name="Qwen/Qwen2-0.5B-Instruct"):
        self.device = "cpu"
        self.data_path = data_path  # Store data path

        # Load SBERT for embeddings
        self.sbert_model = SentenceTransformer(model_name, device=self.device)
        self.sbert_model = self.sbert_model.half()

        # Load Qwen model for text generation
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(self.device)

        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)

        # Load or create FAISS index
        self.load_or_create_index()

    import os  # Import os for file checks

    def load_or_create_index(self):
        """Loads FAISS index and index_map if they exist, otherwise creates new ones."""
        if os.path.exists("financial_faiss.index") and os.path.exists("index_map.txt"):
            try:
                self.faiss_index = faiss.read_index("financial_faiss.index")
                with open("index_map.txt", "r", encoding="utf-8") as f:
                    self.index_map = {i: line.strip() for i, line in enumerate(f)}
                print("FAISS index and index_map loaded successfully.")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Recreating index...")
                self.create_faiss_index()
        else:
            print("FAISS index or index_map not found. Creating a new one...")
            self.create_faiss_index()


    def create_faiss_index(self):
        """Creates a FAISS index from the provided Excel file."""
        df = pd.read_excel(self.data_path)
        sentences = []
        self.index_map = {}  # Initialize index_map

        for row_idx, row in df.iterrows():
            for col in df.columns[1:]:  # Ignore the first column (assumed to be labels)
                sentence = f"{row[df.columns[0]]} - year {col} is: {row[col]}"
                sentences.append(sentence)
                self.index_map[len(self.index_map)] = sentence  # Store mapping

        # Encode the sentences into embeddings
        embeddings = self.sbert_model.encode(sentences, convert_to_numpy=True)

        # Create FAISS index (FlatL2 for simplicity)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        # Save index and index map
        faiss.write_index(self.faiss_index, "financial_faiss.index")
        with open("index_map.txt", "w", encoding="utf-8") as f:
            for sentence in self.index_map.values():
                f.write(sentence + "\n")

    def query_faiss(self, query, top_k=3):
        """Retrieves the top_k closest sentences from FAISS index."""
        query_embedding = self.sbert_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        results = [self.index_map[idx] for idx in indices[0] if idx in self.index_map]
        confidences = [1 - (dist / (np.max(distances[0]) or 1)) for dist in distances[0]]

        return results, confidences

    def moderate_query(self, query):
        """Blocks inappropriate queries containing restricted words."""
        BLOCKED_WORDS = re.compile(r"\b(hack|bypass|illegal|exploit|scam|kill|laundering|murder|suicide|self-harm)\b", re.IGNORECASE)
        return not bool(BLOCKED_WORDS.search(query))

    def generate_answer(self, context, question):
        messages = [
            {"role": "system", "content": "You are a financial assistant. Answer only finance-related questions. If the question is not related to finance, reply: 'I'm sorry, but I can only answer financial-related questions.' If the user greets you (e.g., 'Hello', 'Hi', 'Good morning'), respond politely with 'Hello! How can I assist you today?'."},
            {"role": "user", "content": f"{question} - related contect extracted form db {context}"}
        ]

        # Use Qwen's chat template
        input_text = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and move input to device
        inputs = self.qwen_tokenizer([input_text], return_tensors="pt").to(self.device)
        self.qwen_model.config.pad_token_id = self.qwen_tokenizer.eos_token_id

        # Generate response
        outputs = self.qwen_model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            pad_token_id=self.qwen_tokenizer.eos_token_id,
        )

        # Extract only the newly generated part
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]  # Remove prompt part
        response = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response



    def get_answer(self, query):
        """Main function to process a user query and return an answer."""
        
        # Check if query is appropriate
        if not self.moderate_query(query):
            return "Inappropriate request.", 0.0

        # Retrieve relevant documents and their confidence scores
        retrieved_docs, confidences = self.query_faiss(query)
        if not retrieved_docs:
            return "No relevant information found.", 0.0

        # Combine retrieved documents as context
        context = " ".join(retrieved_docs)
        avg_confidence = round(sum(confidences) / len(confidences), 2)

        # Generate model response
        model_response = self.generate_answer(context, query)

        # Extract only the relevant part of the response
        model_response = model_response.strip()
        
        # Ensure only the actual answer is returned
        if model_response.lower() in ["i don't know", "no relevant information found"]:
            return "I don't know.", avg_confidence
        #print(avg_confidence)
        if avg_confidence == 0.0:
            return "Not relevant ", avg_confidence

        
        return model_response, avg_confidence
