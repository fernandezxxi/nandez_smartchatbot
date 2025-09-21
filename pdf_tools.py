#pip install -q langchain PyPDF2
import os
import io
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document

def load_pdf_agent(pdf_path, google_api_key):
    # Membuka dan membaca file PDF
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_pages = pdf_reader.pages
        context = "\n\n".join(page.extract_text() for page in pdf_pages)

    # Memecah teks menjadi potongan-potongan kecil
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(context)
    documents = [Document(page_content=text) for text in texts]

    # Inisialisasi embeddings dan vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # model khusus embedding
        google_api_key=google_api_key
    )
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Membuat agent QA dengan LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key
    )
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    def ask_pdf_agent(question):
        docs = vectorstore.similarity_search(question, k=3)
        answer = qa_chain.run(input_documents=docs, question=question)
        return answer

    return ask_pdf_agent

# Contoh penggunaan:
if __name__ == "__main__":
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pdf_path = 'bphn.pdf'
    ask_pdf_agent = load_pdf_agent(pdf_path, google_api_key)
    
    pertanyaan = "Apa isi pasal 1 BPHN?"
    jawaban = ask_pdf_agent(pertanyaan)
    print(jawaban)