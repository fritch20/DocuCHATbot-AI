import gradio as gr
from pdf_utils import extract_text_from_pdf, chunk_text
from rag_utils import RAGChatbot


# Instance globale du chatbot
chatbot = RAGChatbot(
    model_name="all-MiniLM-L6-v2",
    ollama_model="llama3"   # tu peux mettre "mistral" si tu préfères
)

document_loaded = {
    "status": False,
    "filename": ""
}


def process_pdf(pdf_file):
    """
    Charge le PDF, extrait le texte, crée les chunks et l'index FAISS.
    """
    if pdf_file is None:
        return "Aucun fichier PDF sélectionné."

    try:
        pdf_path = pdf_file.name

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return "Impossible d'extraire du texte du PDF."

        chunks = chunk_text(text, chunk_size=300, overlap=50)
        if not chunks:
            return "Aucun chunk généré à partir du PDF."

        chatbot.build_index(chunks)

        document_loaded["status"] = True
        document_loaded["filename"] = pdf_path

        return f"PDF chargé avec succès.\n\nNombre de chunks créés : {len(chunks)}"

    except Exception as e:
        return f"Erreur lors du traitement du PDF : {str(e)}"


def ask_question(question):
    """
    Pose une question au chatbot.
    """
    if not document_loaded["status"]:
        return "Veuillez d'abord charger un PDF.", "Aucune source."

    if not question or not question.strip():
        return "Veuillez entrer une question.", "Aucune source."

    try:
        answer, sources = chatbot.ask(question, k=3)
        sources_text = "\n\n---\n\n".join(sources) if sources else "Aucune source retrouvée."
        return answer, sources_text
    except Exception as e:
        return f"Erreur : {str(e)}", "Aucune source."


def summarize_document():
    """
    Résume le document entier à partir des premiers chunks.
    """
    if not document_loaded["status"]:
        return "Veuillez d'abord charger un PDF."

    try:
        selected_chunks = chatbot.chunks[:5]
        prompt = f"""
Tu es un assistant intelligent.
Résume clairement le document suivant en français.
Mets en avant :
- le sujet principal
- les points importants
- les informations essentielles

Document :
{" ".join(selected_chunks)}

Résumé :
"""
        answer = chatbot.call_ollama(prompt)
        return answer
    except Exception as e:
        return f"Erreur lors du résumé : {str(e)}"


with gr.Blocks(title="Chatbot RAG PDF") as demo:
    gr.Markdown("# Chatbot RAG PDF")
    gr.Markdown("Charge un document PDF, puis pose des questions dessus.")

    with gr.Row():
        pdf_input = gr.File(label="Choisir un PDF", file_types=[".pdf"])
        load_button = gr.Button("Charger le PDF")

    status_output = gr.Textbox(label="Statut", lines=4)

    load_button.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status_output
    )

    gr.Markdown("## Poser une question")

    question_input = gr.Textbox(
        label="Ta question",
        placeholder="Exemple : Quels sont les points importants de ce document ?"
    )
    ask_button = gr.Button("Envoyer")

    answer_output = gr.Textbox(label="Réponse du chatbot", lines=8)
    sources_output = gr.Textbox(label="Passages retrouvés", lines=12)

    ask_button.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    gr.Markdown("## Résumé automatique")
    summary_button = gr.Button("Résumer le document")
    summary_output = gr.Textbox(label="Résumé", lines=8)

    summary_button.click(
        fn=summarize_document,
        inputs=[],
        outputs=summary_output
    )


if __name__ == "__main__":
    demo.launch()