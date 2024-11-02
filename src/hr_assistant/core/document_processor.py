""" Document processor core class """

from typing import Any, List

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    A class to load, split, and process PDF documents.

    Attributes:
        text_splitter (RecursiveCharacterTextSplitter): Split documents based on
            specified chunk size, overlap, and separators.

    Methods:
        load_document(data_path: str) -> List[Any]:
            Loads a document from the specified path.

        split_document(document: List[Any]) -> List[Any]:
            Splits the loaded document into chunks using the text splitter.

        remove_newlines(document: List[Any]) -> List[Any]:
            Removes newline characters from each chunk in the document.

        process_document(document: List[Any]) -> List[Any]:
            Splits the document and removes newlines, returning a processed document.
    """

    def __init__(
        self,
        text_splitter_chunk_size: int,
        text_splitter_chunk_overlap: int,
        text_splitter_separators: str,
    ) -> None:
        """
        Initializes the DocumentProcessor with specified text splitting parameters.

        Args:
            text_splitter_chunk_size (int): Maximum size of each chunk of text.
            text_splitter_chunk_overlap (int): Overlap size between chunks.
            text_splitter_separators (str): Separator(s) to use for splitting text.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_chunk_size,
            chunk_overlap=text_splitter_chunk_overlap,
            separators=text_splitter_separators,
            keep_separator=False,
        )

    @staticmethod
    def load_document(data_path: str) -> List[Any]:
        """
        Loads a document from the specified file path.

        Args:
            data_path (str): The file path to the PDF document.

        Returns:
            List[Any]: A list of document chunks loaded from the PDF.
        """
        return PyPDFLoader(data_path).load()

    def split_document(self, document: List[Any]) -> List[Any]:
        """
        Splits the provided document into smaller chunks.

        Args:
            document (List[Any]): A list representing the document to be split.

        Returns:
            List[Any]: A list of document chunks after splitting.
        """
        return self.text_splitter.split_documents(document)

    @staticmethod
    def remove_newlines(document: List[Any]) -> List[Any]:
        """
        Removes newline characters from each chunk in the document.

        Args:
            document (List[Any]): A list of document chunks to process.

        Returns:
            List[Any]: The document with newline characters removed from each chunk.
        """
        for chunk in document:
            chunk.page_content = chunk.page_content.replace("\n", "")
        return document

    def process_document(self, document: List[Any]) -> List[Any]:
        """
        Processes the document by splitting it and removing newlines.

        Args:
            document (List[Any]): A list representing the document to process.

        Returns:
            List[Any]: The processed document with chunks split and newlines removed.
        """
        splitted_document = self.split_document(document=document)
        processed_document = self.remove_newlines(splitted_document)
        return processed_document
