import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted text:
    - Removes boilerplate like 'Modified on:'
    - Collapses excessive newlines and spaces
    """
    cleaned = re.sub(r"Modified on:.*?\n", "", raw_text)
    cleaned = re.sub(r"Figure\s+\d+(\.\w+)?[:]?.*", "", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


# def chunk_text(text: str, max_words: int = 250) -> list:
#     """
#     Splits long text into word-aware, sentence-bound chunks for embeddings.
#     """
#     sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
#     chunks = []
#     current_chunk = ""
#     for sentence in sentences:
#         if len((current_chunk + " " + sentence).split()) <= max_words:
#             current_chunk += " " + sentence
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks


def chunk_text(text: str, max_chars: int = 500, overlap: int = 150) -> list:
    """
    Smart semantic chunking using LangChain RecursiveCharacterTextSplitter.
    - Breaks text based on section headings, paragraphs, then sentences.
    - Prevents breaking mid-clause for legal documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = splitter.create_documents([text])
    return [doc.page_content.strip() for doc in documents]

def table_to_markdown(table):
    """
    Converts pdfplumber list-of-lists table to markdown (optional).
    """
    if not table or not table[0]:
        return ""
    header = "| " + " | ".join(str(c).strip() for c in table[0]) + " |"
    sep = "| " + " | ".join("---" for _ in table[0]) + " |"
    rows = ["| " + " | ".join(str(c).strip() for c in row) + " |" for row in table[1:]]
    return "\n".join([header, sep] + rows)
