import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google import GoogleGenerativeAIEmbeddings
from itertools import chain

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_embeddings():
    return embeddings

# Prompts

summary_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert medical professional tasked with creating a concise yet comprehensive summary of a medical document.
    Your summary must be factually accurate, clinically relevant, and organized in a way that highlights key information.
    SUMMARY FORMAT:
    1. Begin with 1-2 sentence overview of the document's key content
    2. Organize the remaining information into clinically relevant sections
    3. Use concise, clear language
    4. Include specific values, measurements, and dates where provided
    5. Focus more on the dates and mention dates
    6. Total length should be approximately 150-200 words depending on document complexity

    <context>
    {context}
    </context>
    """
)

history_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert medical professional creating a comprehensive summary of a patient's medical history based on document summaries and patient information. Your goal is to produce a medically accurate, chronologically organized narrative that would help a new physician quickly understand this patient's medical journey.

    PATIENT INFORMATION:
    {patient_info}

    DOCUMENT SUMMARIES:
    {context}

    INSTRUCTIONS FOR COMPREHENSIVE SUMMARY:
    1. First, carefully review and extract all information from the PATIENT INFORMATION section. This contains verified demographic details, known conditions, and background that must be accurately represented.
    
    2. Create a chronological timeline of the patient's medical events using date information from document summaries. Sort all documents by their dates (DD/MM/YYYY format) before synthesizing the information.
    
    3. Organize your summary into these mandatory sections:
       - **Demographics and Background**: Accurately state patient's name, age (calculate from DOB), gender, occupation, height, weight, blood group, and lifestyle factors exactly as provided in PATIENT INFORMATION.
       - **Chief Complaints and Present Illness**: Identify the primary medical issues and their progression over time.
       - **Past Medical History**: Include all conditions mentioned in patient information and document summaries.
       - **Medications**: Document all medications mentioned, with dates when they were prescribed or changed.
       - **Allergies**: List all allergies exactly as stated in patient information.
       - **Surgical History and Procedures**: Detail all surgical procedures with exact dates.
       - **Laboratory and Diagnostic Findings**: Organize test results chronologically, noting abnormal values and trends.
       - **Assessment and Diagnoses**: Summarize diagnoses from most recent to earliest.
       - **Treatment Plan and Recommendations**: Include all treatment recommendations with their dates. Do not emphasis much on treatment plans that are old.
       
    4. For each document summary, extract:
       - The document type (e.g., "Blood test Report", "Imaging Report")
       - The exact date in DD/MM/YYYY format
       - All significant findings, results, or clinical notes
       - Any diagnoses, treatments, or recommendations
       
    5. When integrating information:
       - Maintain strict chronological order within each section
       - Include specific dates for all events, tests, and procedures
       - Note significant changes in test results or clinical status over time
       - Preserve medical terminology and specific values/measurements
       - Identify connections between different findings and their clinical significance
       
    6. If information is truly not available for a section despite being in patient information or document summaries, only then state "Not documented."

    7. Use the patient's actual age (calculated from date of birth) and current year (2025).
    
    8. Include a brief conclusion summarizing the patient's current status and key medical concerns.

    COMPREHENSIVE SUMMARY:
    """
)

chatbot_prompt = ChatPromptTemplate.from_template(
    """
    You are a medical chatbot designed to assist users by providing accurate and personalized responses based on a patient's medical records and inquiries. Your responses should be factually correct, concise, and user-friendly.

    PATIENT INFORMATION:
    {patient_info}

    USER QUERY:
    {query}

    DOCUMENTS:
    You have access to the following patient documents:
    {context}

    INSTRUCTIONS FOR RESPONSE:
    1. Carefully review the USER QUERY and identify the specific information being requested.
    2. Search through the DOCUMENTS provided to extract relevant information that addresses the query. Use document metadata (e.g., type, date) to prioritize contextually relevant documents.
    3. Cross-reference the extracted information with PATIENT INFORMATION to ensure the response is personalized and clinically accurate.
    4. Provide a concise, clear answer or recommendation based on the extracted information from the documents.
    5. If the requested information cannot be found in the DOCUMENTS or PATIENT INFORMATION, state "Not documented" or "Unable to provide information at this time."
    
    RESPONSE FORMAT:
    - Begin with a friendly greeting or acknowledgment of the user's inquiry.
    - Provide a concise answer or recommendation based on the context and patient documents.
    - Include specific values, dates, or findings from the documents where applicable.
    - Offer additional resources or suggestions if applicable.

    RESPONSE:
    """
)


def process_uploaded_file(directory_path):
    """Processes a given PDF file and returns extracted documents."""
    loader = PyPDFLoader(directory_path)
    all_documents = loader.load()
    
    return all_documents

def vector_embedding(documents,flag):
    """Generates vector embeddings from documents."""
    print("Processing documents...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if flag:        #for a vector db of all documents
        documents = list(chain.from_iterable(documents))

    final_documents = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(final_documents, embeddings)
    print("Documents processed successfully!")
    return vectors

def generate_summary(vectors):
    """Generates a summary of the uploaded documents."""
    document_chain = create_stuff_documents_chain(llm, summary_prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({'input': 'Summarize the document'})
    
    print("Summary generated in seconds!\n")
    # print("=== DOCUMENT SUMMARY ===")
    # print(response['answer'])
    return response['answer']

def generate_history(vectors,patient_info):
    """Generates a summary of the uploaded documents."""
    document_chain = create_stuff_documents_chain(llm, history_prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({
            'input': 'Generate a comprehensive patient history summary',
            'patient_info': patient_info
        })
    
    print("Summary generated in seconds!\n")
    # print("=== DOCUMENT SUMMARY ===")
    # print(response['answer'])
    return response['answer']
    
def generate_response(vectors,patient_info,query):
    document_chain = create_stuff_documents_chain(llm, chatbot_prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({
            'input': 'Please provide the most accurate response based on the query',
            'patient_info': patient_info,
            'query':query
        })
    # print("=== DOCUMENT SUMMARY ===")
    # print(response['answer'])
    return response['answer']


































# def ask_question(vectors, question):
#     """Answers a user question based on the uploaded documents."""
#     document_chain = create_stuff_documents_chain(llm, qa_prompt)
#     retriever = vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
#     start = time.process_time()
#     response = retrieval_chain.invoke({'input': question})
#     elapsed_time = time.process_time() - start
    
#     print(f"Answer found in {elapsed_time:.2f} seconds!\n")
#     print("=== ANSWER ===")
#     print(response['answer'])




# parser = argparse.ArgumentParser(description="MEDPASS")
    # parser.add_argument("--file", type=str, help="Path to the PDF document")
    # parser.add_argument("--summary", action="store_true", help="Generate a summary of the document")
    # parser.add_argument("--question", type=str, help="Ask a question about the document")
    # args = parser.parse_args()
    
    # if not args.file:
    #     print("Error: Please provide a PDF file using --file")
    #     return
    
    # if not os.path.exists(args.file):
    #     print("Error: File not found!")
    #     return


#   qa_prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     </context>
#     Questions: {input}
#     """
# )
