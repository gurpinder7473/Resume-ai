import pdfminer.high_level
import io

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_data = uploaded_file.read()
        pdf_io = io.BytesIO(pdf_data)
        text = pdfminer.high_level.extract_text(pdf_io)

        # Debugging: Print extracted text to check if education details exist
        print("\n========= Extracted Text from Resume =========\n")
        print(text)
        print("\n===============================================\n")

        return text.strip() if text else "No text found"
    except Exception as e:
        return f"Error extracting text: {str(e)}"
