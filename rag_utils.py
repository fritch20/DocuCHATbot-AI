import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


class RAGChatbot:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", ollama_model: str = "llama3"):
        """
        model_name : modèle d'embeddings
        ollama_model : modèle local Ollama
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.ollama_model = ollama_model

        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks: list):
        """
        Construit l'index FAISS à partir des chunks.
        """
        if not chunks:
            raise ValueError("Aucun chunk à indexer.")

        self.chunks = chunks
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        self.index = index
        self.embeddings = embeddings

    def search(self, query: str, k: int = 3):
        """
        Recherche les chunks les plus proches de la question.
        """
        if self.index is None:
            raise ValueError("L'index n'est pas encore construit.")

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i in indices[0]:
            if 0 <= i < len(self.chunks):
                results.append(self.chunks[i])

        return results

    def build_prompt(self, context_chunks: list, question: str) -> str:
        """
        Construit le prompt envoyé au LLM.
        """
        context = "\n\n".join(context_chunks)

        prompt = f"""
Tu es un assistant intelligent spécialisé dans l'analyse de documents.

Règles :
- Réponds uniquement à partir du contexte fourni.
- Si l'information n'est pas dans le contexte, dis clairement : "Je ne trouve pas cette information dans le document."
- Réponds de manière claire, structurée et concise.
- Réponds en français.

Contexte :
{context}

Question :
{question}

Réponse :
"""
        return prompt.strip()

    def call_ollama(self, prompt: str) -> str:
        """
        Appelle Ollama en local via son API HTTP.
        """
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Aucune réponse générée.")
        except requests.exceptions.RequestException as e:
            return f"Erreur lors de l'appel à Ollama : {str(e)}"

    def ask(self, question: str, k: int = 3):
        """
        Pipeline complet :
        1. recherche des chunks pertinents
        2. construction du prompt
        3. génération de la réponse  
        """
        context_chunks = self.search(question, k=k)
        prompt = self.build_prompt(context_chunks, question)
        answer = self.call_ollama(prompt)

        return answer, context_chunks