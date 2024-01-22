import os
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import joblib


PATH = os.path.dirname(os.path.abspath(__file__))
BOOKS_PATH = os.path.join(PATH, "books")

# File to store/read embeddings from the context
EMBEDDINGS_DIR = os.path.join(PATH, "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "book.embeddings")



def create_embeddings(context_file_path, embeddings):
    # Read context
    pages = []
    for root, _, files in os.walk(context_file_path):
        for name in files:
            book_path = os.path.join(root, name)
            print(book_path) 
            loader = PyPDFLoader(book_path)
            tmp_pages = loader.load_and_split()
            for i, page in enumerate(tmp_pages):
                page.page_content= "Book title: " + name.replace(".pdf", "") + "\nPage: " + str(i) + "\n\n" + page.page_content.replace("More books on http://adapted -english -books.site", "")
            pages.extend(tmp_pages)
            #print(pages[0])
            
    # Save on disk
    db = FAISS.from_documents(pages, embeddings)
    saved_db = db.serialize_to_bytes()
    joblib.dump(saved_db, EMBEDDINGS_FILE)


def load_embeddings(file_path, embeddings_gen):
    saved_db = joblib.load(file_path)
    db = FAISS.deserialize_from_bytes(
        embeddings=embeddings_gen, serialized=saved_db) 
    return db


def create_message(examples, question):
    message = """
    # IDENTITY
    You are a scholar specialized in J. R. R. Tolkien's 'Lord of the Rings' and 'The Hobbit'
    
    # TASK
    You task is to answer questions about The Hobbit and Lord of the Rings
    Explain where did you get the response from
    You can only respond based on the information from the RELEVANT PAGES below.
    You're roleplaying as Gandalf, an old and wise wizard from those books, so you have to address the user in an ancient and magical english. 

    # RELEVANT PAGES TO ANSWER THE QUESTION
    Question: {examples}

    # QUESTION
    Answer: {question}
    
    """.format(examples=examples, question=question)
    return message


def get_examples(query, db):
    docs = db.similarity_search_with_score(query, k=2)
    examples = ""
    for doc in docs:
        examples = examples + "\n\n" + doc[0].page_content
    return examples


if __name__ == "__main__":

    # Create folders if needed
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Initialize LLM
    llm = VertexAI(model_name="text-bison-32k", max_output_tokens=500, temperature=0.5)

    # Initialize embedding generator
    embeddings_gen = VertexAIEmbeddings(model_name="textembedding-gecko@001")

    # Create embeddings if they do not exist
    if not os.path.isfile(EMBEDDINGS_FILE):
        create_embeddings(BOOKS_PATH, embeddings_gen)

    # Load embeddedings
    db = load_embeddings(EMBEDDINGS_FILE, embeddings_gen)

    # Load file to translate
    while True:
        query = input("Question: ")
        print("Question: ", query)

        # RAG with FAISS
        examples = get_examples(query, db) 

        # Create system message and context
        message = create_message(examples, query)
        print(examples, "\n\n\n------------------------")

        # Call Palm2 API
        text = llm(message)
        print("Response: ", text, "\n")