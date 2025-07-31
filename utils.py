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

from transformers import GPT2TokenizerFast

# Load tokenizer only once
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# def chunk_text(text: str, max_tokens: int = 300, overlap: int = 100) -> list:
#     """
#     Token-aware chunking for better GPT alignment.
#     - max_tokens: Max tokens per chunk (300 recommended)
#     - overlap: Token overlap between chunks (100 recommended)
#     """
#     tokens = tokenizer.encode(text)
#     chunks = []
#     start = 0

#     while start < len(tokens):
#         end = min(start + max_tokens, len(tokens))
#         chunk = tokenizer.decode(tokens[start:end])
#         chunks.append(chunk.strip())
#         start += max_tokens - overlap

#     return chunks


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

