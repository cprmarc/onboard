import glob
from pathlib import Path

# Mappa, ahová feltöltöd az onboarding anyagokat
DOCUMENT_DIR = "documents/"

# Dokumentumok automatikus beolvasása
documents = []
for filepath in glob.glob(f"{DOCUMENT_DIR}/*"):
    path = Path(filepath)
    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix == ".txt":
        loader = TextLoader(str(path))
    elif path.suffix == ".html":
        loader = UnstructuredHTMLLoader(str(path))
    else:
        continue
    documents.extend(loader.load())

# A további lépések (splitter, embeddings, retriever, stb.) változatlanul maradnak
