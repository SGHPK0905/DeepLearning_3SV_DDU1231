"""
BTL02: PDF Exporter via Microsoft Word and PowerPoint COM automation
Converts .docx to .pdf and .pptx to .pdf natively.
"""

import os
from pathlib import Path
import comtypes.client


def export_docx_to_pdf(docx_path, pdf_path):
    docx_path = Path(docx_path).resolve()
    pdf_path = Path(pdf_path).resolve()

    print(f"Exporting Word to PDF: {docx_path.name} -> {pdf_path.name}...")
    word = None
    doc = None
    try:
        word = comtypes.client.CreateObject('Word.Application')
        word.Visible = False
        doc = word.Documents.Open(str(docx_path))
        # 17 = wdFormatPDF
        doc.SaveAs(str(pdf_path), FileFormat=17)
        print(f"Exported successfully: {pdf_path}")
        return True
    except Exception as e:
        print(f"Error exporting Word to PDF: {e}")
        return False
    finally:
        if doc is not None:
            doc.Close()
        if word is not None:
            word.Quit()


def export_pptx_to_pdf(pptx_path, pdf_path):
    pptx_path = Path(pptx_path).resolve()
    pdf_path = Path(pdf_path).resolve()

    print(f"Exporting PowerPoint to PDF: {pptx_path.name} -> {pdf_path.name}...")
    powerpoint = None
    deck = None
    try:
        powerpoint = comtypes.client.CreateObject('PowerPoint.Application')
        # 32 = ppSaveAsPDF
        deck = powerpoint.Presentations.Open(str(pptx_path), WithWindow=False)
        deck.SaveAs(str(pdf_path), FileFormat=32)
        print(f"Exported successfully: {pdf_path}")
        return True
    except Exception as e:
        print(f"Error exporting PowerPoint to PDF: {e}")
        return False
    finally:
        if deck is not None:
            deck.Close()
        if powerpoint is not None:
            powerpoint.Quit()
