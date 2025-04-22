"""
rag_app: A FastAPI service that ingests medical PDFs, creates/upserts
FAISS embeddings in Firestore, and serves both summaries and chatbot
responses via LangChain.

Key endpoints:
- POST  /         : ingest + summarize new docs, update history/all vector stores
- POST  /chatbot  : query the vector store to generate an answer
- POST  /compress : compress + store uploaded PDF metadata
"""
import asyncio
from collections import defaultdict
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
from app import process_uploaded_file,vector_embedding,generate_summary,generate_history,generate_response,get_embeddings
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
import requests
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from shutil import rmtree
from compression import compress_pdf
from datetime import datetime


user_locks = defaultdict(asyncio.Lock)

rag_app = FastAPI()

rag_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = get_embeddings()


cred  = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred , {"databaseURL": "https://medpass-try-default-rtdb.firebaseio.com/","storageBucket":"medpass-try.firebasestorage.app"})

bucket = storage.bucket()

def download_pdf(pdf_url):
    """
    download pdf document and return it in bytes form

    input - pdf_url
    output - Bytes of pdf
    """
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download pdf. Status code {response.status_code}")

db = firestore.client() 



class User(BaseModel):
    username : str
    info : dict
    delete_doc : bool


def load_embeddings(save_path,vectors_doc : dict):
    """
    Load a FAISS vector store from URLs of firebase storage and deserialize it locally.

    Args:
        save_path (str): Local directory path where downloaded FAISS files are temporarily stored.
        vectors_doc (dict): Dictionary containing public URLs to 'index.faiss' and 'index.pkl'.

    Returns:
        FAISS: A FAISS vector store object ready for retrieval tasks.

    Side Effects:
        - Downloads files from firebase storage.
        - Creates and deletes a local temp directory (`save_path`).
    """
    
    faiss_url = vectors_doc['index.faiss']
    pkl_url = vectors_doc['index.pkl']
    print(faiss_url)

    faiss_doc = download_pdf(faiss_url)
    pkl_doc = download_pdf(pkl_url)

    os.mkdir(save_path)
    with open(os.path.join(save_path,"index.faiss"), 'wb') as file:
        file.write(faiss_doc)
    with open(os.path.join(save_path,"index.pkl"), 'wb') as file:
        file.write(pkl_doc) 

    vectors = FAISS.load_local(
            folder_path = save_path,
            embeddings = embeddings,
            allow_dangerous_deserialization=True
        )
    rmtree(save_path)
    return vectors
    

@rag_app.post("/")
async def summary(user: User = None):   
    
    
    username = user.username
    patient_info = user.info        # user's info in dict
    delete_doc = user.delete_doc    # see if user has deleted and doc
    async with user_locks[username]:
        print("request recieved............................................................................")
        print(username)
        print(patient_info)

        all_docs = []  # for chatbot
        history = []    #for summary
        page = 1
        count = 0 

        # vector embedding of patient health summary
        his_vec_path = "history_vectordb"       # path for locally storing the the history vector embeddings

        # vector embedding of all the documents
        all_vec_path = "all_vectordb"           # path for locally storing the the all vector embeddings



        # check if the history embeddings already exists
        his_vec_ref = db.collection(username).document("histor_vector_embedding")
        his_vec_ref_snapshot = his_vec_ref.get()

        history_vectors = None
        # if they exist then download them and load the vector db 
        # skip if any doc is deleted
        if his_vec_ref_snapshot.exists and not delete_doc:
            history_vectors = load_embeddings(his_vec_path,his_vec_ref_snapshot.to_dict())

        all_vec_ref = db.collection(username).document("all_vector_embedding")
        all_vec_ref_snapshot = all_vec_ref.get()

        all_doc_vectors = None
        # if they exist then download them and load the vector db 
        # skip if any doc is deleted
        if his_vec_ref_snapshot.exists and not delete_doc:
            all_doc_vectors = load_embeddings(all_vec_path,all_vec_ref_snapshot.to_dict())
            
            

        files = db.collection(username).stream()   # create stream for accessing the files

        if not files:
            return {"no documents"}
        for file in files:
            file_dict = file.to_dict()
            # skip these documents
            if file.id == "#&_summary_&#" or file.id == "histor_vector_embedding" or file.id == "all_vector_embedding":
                continue
            print(file.id)

            # check if already embedded and uploaded to the vector db
            if "embedded" in file_dict and file_dict["embedded"] == 'true' and not delete_doc:
                continue
            count += 1
            file_path = file_dict['fileURL']
            print(file_path)
            pdf = download_pdf(file_path)

            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf)
                temp_file_path = temp_file.name
            documents = process_uploaded_file(temp_file_path) 
            os.remove(temp_file_path)
            all_docs.append(documents)
            curr_vectors = vector_embedding(documents,flag=False)
            doc = Document(page_content=file_dict['issuedOn'] + '\n' + generate_summary(curr_vectors),metadata={"page": page})
            page+=1
            history.append(doc)

            db.collection(username).document(file.id).update({"embedded" : "true"})
        if count == 0:
            return {"no documents"}
        if not history_vectors :
            history_vectors = vector_embedding(history,flag=False)
        else:
            if history: 
                history_vectors.add_documents(history)

        if history_vectors:
            history_vectors.save_local(his_vec_path)

            urls = []
            for db_file in os.listdir(his_vec_path):
                blob = bucket.blob(f"{username}/his_{db_file}")
                blob.upload_from_filename(os.path.join(his_vec_path,db_file))
                blob.make_public()
                urls.append(blob.public_url)
            
            db.collection(username).document("histor_vector_embedding").set({"index.faiss" : urls[0],"index.pkl" : urls[1]})

            rmtree(his_vec_path)

        if not all_doc_vectors :
            all_doc_vectors = vector_embedding(all_docs,flag=True)
        else:
            if all_docs:
                for new_doc in all_docs:
                    all_doc_vectors.add_documents(new_doc)
        
        if all_doc_vectors:
            all_doc_vectors.save_local(all_vec_path)
            urls = []
            for db_file in os.listdir(all_vec_path):
                blob = bucket.blob(f"{username}/all_{db_file}")
                blob.upload_from_filename(os.path.join(all_vec_path,db_file))
                blob.make_public()
                urls.append(blob.public_url)
            
            db.collection(username).document("all_vector_embedding").set({"index.faiss" : urls[0],"index.pkl" : urls[1]})

            rmtree(all_vec_path)

        if history_vectors:
            generated_history = generate_history(history_vectors,patient_info)

            db.collection(username).document("#&_summary_&#").set({"summary" : generated_history})

            return {"summary generated"}
        return {"no documents"}

class Chat(BaseModel):
    question : str
    username : str
    info : dict
    

@rag_app.post("/chatbot")
def chat(chat : Chat = None):
    all_vec_path = "all_vectordb" 
    
    all_vec_ref = db.collection(chat.username).document("all_vector_embedding")
    all_vec_ref_snapshot = all_vec_ref.get()

    all_doc_vectors = None
    # if they exist then download them and load the vector db 
    if all_vec_ref_snapshot.exists:
        all_doc_vectors = load_embeddings(all_vec_path,all_vec_ref_snapshot.to_dict())
    
    answer = generate_response(all_doc_vectors,chat.info,chat.question)

    return {"response": answer}


@rag_app.post("/compress")
async def store_document(
    username: str = Form(...),
    doctorName: str = Form(...),
    description: str = Form(...),
    illness: str = Form(...),
    fileType: str = Form(...),
    issuedDate: str = Form(...),
    file: UploadFile = File(...),
):
    # 1. Save uploaded file
    file_bytes = await file.read()
    with open("temp_pdf.pdf", "wb") as f:
        f.write(file_bytes)
    
    # 2. Compress it
    compress_pdf("temp_pdf.pdf", "temp_out.pdf", 3)
    os.remove("temp_pdf.pdf")

    # 3. Upload compressed file to GCS
    blob = bucket.blob(f"{username}/{file.filename}")
    with open("temp_out.pdf", "rb") as f:
        blob.upload_from_file(f, content_type=file.content_type)
    blob.make_public()
    os.remove("temp_out.pdf")
    file_url = blob.public_url

    # 4. Store metadata in Firestore
    db.collection(username).document(file.filename).set({
        "doctorName": doctorName,
        "description": description,
        "illness": illness,
        "fileName": file.filename,
        "fileURL": file_url,
        "issuedOn": issuedDate,
        "fileType": fileType,
        "uploadedAt": datetime.now().isoformat()
    })

    # 5. Return success response
    return {
        "message": "File and data uploaded successfully!"
    }

