# rag_pipeline/loader.py

import fitz  # PyMuPDF
from PIL import Image
import os
import io
# from paddleocr import PaddleOCR
import easyocr
import numpy as np

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_pdf_ocr(pdf_path):
    # ocr = PaddleOCR(use_angle_cls=True, lang='korean')
    reader = easyocr.Reader(['ko', 'en'])
    doc = fitz.open(pdf_path)
    results = []

    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_result = reader.readtext(np.array(img))
        page_text = "\n".join([line[1] for line in ocr_result])
        results.append(page_text)

    return "\n".join(results)

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append((img, {"source": pdf_path, "page": i + 1}))
    return images

def load_documents(mode="text"):
    docs = []
    for fname in os.listdir("docs"):
        if not fname.endswith(".pdf"):
            continue
        path = os.path.join("docs", fname)
        if mode == "text":
            content = extract_text_from_pdf(path)
        else:
            content = extract_text_from_pdf_ocr(path)
        docs.append({"content": content, "meta": {"source": fname}})
    return docs
