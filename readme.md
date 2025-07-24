# PDF-QA: General PDF Q&A Chatbot

This project is a Streamlit-based web application that allows users to upload PDF documents and ask questions about their content. The chatbot answers strictly based on the information found in the uploaded PDFs.

**Note:** This app works for text-based PDFs only, using [PyPDF2](https://pypi.org/project/PyPDF2/) for text extraction. Scanned/image-based PDFs are not supported.

## Features

- Upload one or more text-based PDF files.
- Extracts and displays the text content from PDFs.
- Ask questions in a chat interface.
- Answers are generated using an LLM (OpenAI GPT) and are strictly based on the uploaded PDFs.
- Dummy agent fallback for development without an API key.

## Getting Started

### Prerequisites

- [OpenAI API Key](https://platform.openai.com/account/api-keys) (for production use)

### Deploying on Streamlit Community Cloud

You can deploy this app for free using [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repository to your GitHub account.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in.
3. Click "New app", select your repo and branch, and set `app.py` as the main file.
4. Add your `OPENAI_API_KEY` as a secret in the app's settings (if needed).

No local installation is required!

**Benefit:** You can quickly share your app with others for testing or to let them take it for a spin—just send them your app link!  
*Note: Apps on Streamlit Community Cloud may go to sleep if not used for a while, but they will automatically wake up when visited again.*

### Running Locally

1. Clone this repository:

    ```sh
    git clone <your-repo-url>
    cd PDF-QA
    ```

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. (Optional) Set your OpenAI API key:

    ```sh
    export OPENAI_API_KEY=your-openai-api-key
    ```

### Running the App

Start the Streamlit app:

```sh
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

## Usage

1. Upload one or more PDF files using the uploader.
2. View the extracted PDF content (optional).
3. Ask questions in the chat input box.
4. The chatbot will answer based only on the content of the uploaded PDFs.

## Development

This project includes a `.devcontainer` for use with VS Code Dev Containers or GitHub Codespaces.

## File Structure

- `app.py` — Main Streamlit application.
- `agent_factory.py` — Agent creation logic and system prompt.
- `requirements.txt` — Python dependencies.
- `.devcontainer/` — Dev container configuration.

## License

MIT License

---

*This project uses [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), [Streamlit](https://streamlit.io/), and [PyPDF2](https://pypi.org/project/PyPDF2/)