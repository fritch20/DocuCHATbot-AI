from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrait tout le texte d'un fichier PDF.
    """
    text = ""
    reader = PdfReader(pdf_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Découpe le texte en morceaux avec chevauchement.
    chunk_size = nombre de mots par chunk
    overlap = nombre de mots répétés entre chunks
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks