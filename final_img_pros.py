from img2table.ocr import TesseractOCR
from img2table.document import PDF
from langchain_community.document_loaders import PyPDFLoader


def processing(file_path):
    ocr = TesseractOCR(n_threads=1, lang="eng")

    pdf_path = file_path
    data_dict={}

    doc = PDF(src=pdf_path)

    tables_by_page = doc.extract_tables(ocr=ocr,
                        implicit_rows=True,
                        borderless_tables=True,
                        min_confidence=20)
    for page,tables in tables_by_page.items():
        for idx, table in enumerate(tables,start=1):
            table_dict = {row: [cell.value for cell in cells] for row, cells in table.content.items()}
            data_dict[f"page {page+1}"] = table_dict

    return data_dict

def process_uploaded_file(directory_path):
    """Processes a given PDF file and returns extracted documents."""
    loader = PyPDFLoader(directory_path)
    all_documents = loader.load()
    
    return all_documents

#Example usage:
if __name__ == "__main__":
    file_path = "VG_med_history/descriptive_format/Vaibhav_Gatne_2022_04_15.pdf"
    #file_path = "VG_med_history/table_format/Vaibhav_Gatne_2022_03_11.pdf"
    extracted_data = processing(file_path)
    #extracted_data = process_uploaded_file(file_path)
    print(extracted_data)
