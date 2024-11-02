""" Main application """

import gradio as gr

from hr_assistant.core.config import load_config
from hr_assistant.core.document_processor import DocumentProcessor
from hr_assistant.core.information_retriever import InformationRetriever


def main() -> None:
    """
    Main function to run the HR Assistant application.

    This function loads configuration parameters, initializes document processing
    and information retrieval components, and launches a Gradio interface to handle
    user queries about the document.

    Workflow:
    1. Loads configuration from the specified YAML file.
    2. Processes the document by loading and splitting it into chunks.
    3. Initializes the information retriever with the processed document.
    4. Launches a Gradio interface for interactive querying.

    Returns:
        None
    """
    params = load_config(config_path="configs/params.yml")

    # Initialize DocumentProcessor with parameters from the config
    document_processor = DocumentProcessor(
        text_splitter_chunk_size=params["text_splitter_chunk_size"],
        text_splitter_chunk_overlap=params["text_splitter_chunk_overlap"],
        text_splitter_separators=params["text_splitter_separators"],
    )

    # Load and process the document
    document = document_processor.load_document(params["document_path"])
    processed_document = document_processor.process_document(document=document)

    # Initialize InformationRetriever with processed document
    information_retriever = InformationRetriever(
        llm_model_name=params["llm_model_name"],
        documents=processed_document,
        embedding_path=params["embedding_path"],
    )

    # Set up the Gradio interface
    demo = gr.Interface(
        fn=information_retriever.retrieve_information,
        inputs=gr.Textbox(label="Question", placeholder="Type your question here..."),
        outputs=gr.Textbox(label="Response"),
        title="HR Assistant",
    )
    demo.launch()


if __name__ == "__main__":
    main()
