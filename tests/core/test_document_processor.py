""" Test fort document processor core class """
import pytest

from hr_assistant.core.document_processor import DocumentProcessor


@pytest.fixture(name="document_processor")
def document_processor_fixture():
    """Fixture to create an instance of DocumentProcessor for testing."""
    return DocumentProcessor(
        text_splitter_chunk_size=100,
        text_splitter_chunk_overlap=10,
        text_splitter_separators=",",
    )


def test_load_document(mocker, document_processor):
    """Test loading a PDF document."""
    mock_pdf_loader = mocker.patch("hr_assistant.core.document_processor.PyPDFLoader")

    mock_pdf_loader.return_value.load.return_value = [
        {"page_content": "This is a test content."}
    ]

    document = document_processor.load_document("path/to/test.pdf")

    assert len(document) == 1
    assert document[0]["page_content"] == "This is a test content."
    mock_pdf_loader.assert_called_once_with("path/to/test.pdf")


def test_process_document(mocker, document_processor):
    """Test the complete document processing workflow."""
    mock_document = [{"page_content": "This is a test document."}]

    with mocker.patch.object(
        document_processor,
        "split_document",
        return_value=[{"page_content": "This is a test document."}],
    ):
        with mocker.patch.object(
            document_processor,
            "remove_newlines",
            return_value=[{"page_content": "This is a test document."}],
        ):
            processed_document = document_processor.process_document(mock_document)

            assert len(processed_document) == 1
            assert processed_document[0]["page_content"] == "This is a test document."
