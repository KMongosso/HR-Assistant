from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(
            self,
            text_splitter_chunk_size: str,
            text_splitter_chunk_overlap: str,
            text_splitter_separators: str,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_chunk_size,
            chunk_overlap=text_splitter_chunk_overlap,
            separators=text_splitter_separators,
            keep_separator=False
        )

    @staticmethod
    def load_document(data_path: str):
        return PyPDFLoader(data_path).load()

    def split_document(self, document):
        return self.text_splitter.split_documents(document)

    @staticmethod
    def remove_newlines(document):
        for chunk in document:
            chunk.page_content = chunk.page_content.replace("\n", "")

        return document

    def process_document(self, document):
        splitted_document = self.split_document(document=document)
        processed_document = self.remove_newlines(splitted_document)

        return processed_document