#pip install -qU langchain-community pypdf
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

loader = PyPDFLoader("kuhp.pdf")
pages = loader.load_and_split()

#pip install -U -q nltk
import nltk
nltk.download('punkt')
from langchain_text_splitters import NLTKTextSplitter, RecursiveCharacterTextSplitter

# Misalkan kamu memiliki dokumen seperti berikut
simple_doc = """
halo nama saya sardi irfansyah, saya lahir di jakarta.
Saya irfan. tinggal di Jakarta. saya sangat suka belajar AI terutama di bidang NLP dan CV
"""

print('panjang total karakter:',len(simple_doc),'\n')

# Membuat objek NLTKTextSplitter dengan ukuran chunk dan overlap
text_splitter = NLTKTextSplitter(separator='\n\n',
                                 chunk_size=50,
                                 chunk_overlap=20) #default separator='\n\n'
#chunk 1 jumlahnya 50 char
#chunk 2 jumlahnya 37 char

# Memecah dokumen menjadi beberapa chunk
chunks = text_splitter.split_text(simple_doc)
print(chunks,'\n')

# Menampilkan hasil chunk
for i, chunk in enumerate(chunks):
    #panjang karakter
    print(f"Panjang chunk {i+1}: {len(chunk)} karakter")
    print(f"Chunk {i+1}:")
    print(chunk)
    print("-" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20,
    is_separator_regex=False,
    keep_separator=False
)    

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(pages)