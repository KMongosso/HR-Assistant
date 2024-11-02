""" Test config core functions """
import pytest

from hr_assistant.core.information_retriever import InformationRetriever


@pytest.fixture(name="information_retriever")
def information_retriever_fixture():
    """Fixture to create an instance of InformationRetriever for testing."""
    mock_documents = [{"page_content": "This is a test document."}]
    return InformationRetriever(
        llm_model_name="gpt-3",
        documents=mock_documents,
        embedding_path="path/to/embedding",
    )


def test_information_retriever_initialization(mocker):
    """Test initialization of InformationRetriever."""
    mock_chat_openai = mocker.patch(
        "hr_assistant.core.information_retriever.ChatOpenAI"
    )
    mock_chroma = mocker.patch("hr_assistant.core.information_retriever.Chroma")
    mock_retrieval_qa = mocker.patch(
        "hr_assistant.core.information_retriever.RetrievalQA"
    )
    mock_documents = [{"page_content": "This is a test document."}]

    # Mock Chroma's from_documents method to return a mock retriever
    mock_retrieval_qa.from_chain_type.return_value = mocker.MagicMock()

    retriever = InformationRetriever(
        llm_model_name="gpt-3",
        documents=mock_documents,
        embedding_path="path/to/embedding",
    )

    # Assert that the retriever is initialized correctly
    assert isinstance(retriever.retriever, mocker.MagicMock)
    mock_chat_openai.assert_called_once_with(model_name="gpt-3")
    mock_chroma.from_documents.assert_called_once()
