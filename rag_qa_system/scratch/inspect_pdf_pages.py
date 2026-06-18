import fitz
import os

pdf_path = "d:/Projects/RAG System/rag_qa_system/backend/AWS Customer Agreement.pdf"
if not os.path.exists(pdf_path):
    # Try searching for any pdf in the directory
    import glob
    files = glob.glob("d:/Projects/RAG System/**/*.pdf", recursive=True)
    if files:
        pdf_path = files[0]
    else:
        pdf_path = None

if pdf_path:
    print("Reading PDF:", pdf_path)
    doc = fitz.open(pdf_path)
    print("Total pages:", len(doc))
    # Let's search for "interest" on each page
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if "interest" in text.lower():
            print(f"--- Interest found on Page {i+1} ---")
            print(text[:1500])
            print("="*60)
else:
    print("PDF not found!")
