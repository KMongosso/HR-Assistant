from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


class InformationRetriever:
    def __init__(self, llm_model_name, documents, embedding_path):
        self.retriever = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=llm_model_name),
            retriever=Chroma.from_documents(
                documents=processed_document,
                embedding=OpenAIEmbeddings(),
                persist_directory=embedding_path,
            ),
            return_source_documents=True,
        )

    def retrieve_information(self, query):
        return self.retriever(query)
