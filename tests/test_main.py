""" Test main app """
from hr_assistant.main import main


def test_main(mocker):
    """Test the main function of the HR Assistant application."""
    mock_information_retriever = mocker.patch("hr_assistant.main.InformationRetriever")
    mock_document_processor = mocker.patch("hr_assistant.main.DocumentProcessor")
    mock_load_config = mocker.patch("hr_assistant.main.load_config")

    # Mock configuration parameters
    mock_load_config.return_value = {
        "text_splitter_chunk_size": 100,
        "text_splitter_chunk_overlap": 10,
        "text_splitter_separators": ",",
        "document_path": "path/to/document.pdf",
        "llm_model_name": "gpt-3",
        "embedding_path": "path/to/embedding",
    }

    # Mock the behavior of DocumentProcessor
    mock_processor_instance = mocker.MagicMock()
    mock_document_processor.return_value = mock_processor_instance
    mock_processor_instance.load_document.return_value = [
        {"page_content": "This is a test document."}
    ]
    mock_processor_instance.process_document.return_value = [
        {"page_content": "This is a test document."}
    ]

    # Mock the behavior of InformationRetriever
    mock_information_instance = mocker.MagicMock()
    mock_information_retriever.return_value = mock_information_instance

    # Run the main function
    main()

    # Assertions
    mock_load_config.assert_called_once_with(config_path="configs/params.yml")
    mock_document_processor.assert_called_once_with(
        text_splitter_chunk_size=100,
        text_splitter_chunk_overlap=10,
        text_splitter_separators=",",
    )
    mock_processor_instance.load_document.assert_called_once_with(
        "path/to/document.pdf"
    )
    mock_processor_instance.process_document.assert_called_once_with(
        document=[{"page_content": "This is a test document."}]
    )
    mock_information_retriever.assert_called_once_with(
        llm_model_name="gpt-3",
        documents=[{"page_content": "This is a test document."}],
        embedding_path="path/to/embedding",
    )
