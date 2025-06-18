# Chat With Document - RAG Project

A powerful Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and have interactive conversations with their content using Google's Gemini model.

![Chat With Document Demo](/ScreenShot/image.png)

## 🚀 Features

- Upload and process PDF documents
- Interactive chat interface via Streamlit
- Smart document chunking and vector indexing
- Context-aware responses using RAG architecture
- Streaming responses for better user experience

## 💻 Tech Stack

- **Frontend**: Streamlit
- **Language Model**: Google Generative AI (Gemini-2.0-Flash)
- **Embedding Model**: Text-Embedding-004
- **Vector Database**: FAISS
- **Document Processing**: PyMuPDF, LangChain
- **Environment**: Python, dotenv

## 📋 Prerequisites

- Python 3.8+
- Google API Key for Gemini access
- Basic understanding of virtual environments

## 🔧 Installation & Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chat-with-document.git
    cd chat-with-document
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory with the following variables:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    MODEL=gemini-2.0-flash
    Temperature=0.1
    Num_predict=100
    BASE_URL=your_base_url_if_needed
    ```

5. Run the application:
    ```bash
    streamlit run app.py
    ```

## 🔄 How It Works

1. **Document Upload**: User uploads a PDF document
2. **Processing Pipeline**:
    - PDF text extraction using PyMuPDF
    - Text chunking with RecursiveCharacterTextSplitter
    - Embedding generation with Google's text-embedding-004
    - Vector storage in FAISS
3. **Query Processing**:
    - User asks questions in the chat interface
    - Most relevant document chunks are retrieved using similarity search
    - Context and question are sent to Gemini LLM
    - Response is streamed back to the user

## 🧠 What I Learned

- Implementation of Retrieval-Augmented Generation (RAG) systems
- Working with LangChain for building robust LLM applications
- Vector search optimization with FAISS
- Efficient document processing and chunking strategies
- Integrating Google's Generative AI models
- Building interactive UIs with Streamlit

## 📚 Future Improvements

- Support for additional document types (DOCX, TXT, etc.)
- Multi-document conversations
- User authentication and document history
- Improved context handling for longer documents
- Citation of sources in responses

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- LangChain for their excellent document processing tools
- Google Generative AI team for Gemini model access
- FAISS team for the efficient vector database implementation
- Streamlit for making UI development easy and intuitive