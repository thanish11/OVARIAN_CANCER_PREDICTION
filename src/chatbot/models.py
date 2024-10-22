from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import numpy as np

class OvarianCancerChatbot:
    hospital_store = None
    diet_store = None
    common_store = None
    def __init__(self):
        # Initialize embeddings and language model
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-v0.1",
            model_kwargs={"temperature": 0.7, "max_length": 500},
            huggingfacehub_api_token="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"
        )

        # Load PDFs into vector stores if they haven't been loaded yet
        if OvarianCancerChatbot.hospital_store is None:
            OvarianCancerChatbot.hospital_store = self.load_pdf_vector_store("data/Hospitals.pdf")
        if OvarianCancerChatbot.diet_store is None:
            OvarianCancerChatbot.diet_store = self.load_pdf_vector_store("data/Diet_plan.pdf")
        if OvarianCancerChatbot.common_store is None:
            OvarianCancerChatbot.common_store = self.load_pdf_vector_store("data/merged.pdf")

    def load_pdf_vector_store(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        loaded_pages = loader.load()

        print(f"Loaded {len(loaded_pages)} pages from {pdf_path}")

        if not loaded_pages:
            print(f"No pages found in {pdf_path}")
            return FAISS.from_documents([], self.embeddings)

        documents = []
        for i, page in enumerate(loaded_pages):
            # Ensure the page content is a string
            content = page.text if hasattr(page, 'text') else str(page)
            if content.strip():  # Only add non-empty pages
                documents.append(Document(page_content=content))

        if not documents:
            print("No valid documents to create embeddings.")
            return FAISS.from_documents([], self.embeddings)

        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            print("VectorStore created successfully!")
            return vectorstore
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")
            return FAISS.from_documents([], self.embeddings)

    def calculate_roma_score(self, ca125, he4, menopause_status):
        if menopause_status == "postmenopause":
            return np.exp(-12.0 + 2.38 * np.log(he4) + 0.0626 * np.log(ca125)) / (
                    1 + np.exp(-12.0 + 2.38 * np.log(he4) + 0.0626 * np.log(ca125))
            ) * 100
        else:  # Premenopause
            return np.exp(-8.09 + 1.04 * np.log(he4) + 0.732 * np.log(ca125)) / (
                    1 + np.exp(-8.09 + 1.04 * np.log(he4) + 0.732 * np.log(ca125))
            ) * 100

    def query_pdf(self, store, query):
        docs = store.similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in docs])

    def answer_general_query(self, query):
        response = self.llm(query)
        return response
