# Bajaj Hackathon Project

## 🚀 Project Overview
This project is designed as a modular Retrieval-Augmented Generation (RAG) system, featuring a FastAPI backend, a Streamlit frontend, and benchmarking tools. It enables users to query documents (such as policies) using advanced LLMs, with a focus on performance, flexibility, and ease of use.

## 🛠️ Features & Workflow
- **Document Ingestion**: Upload and process documents (e.g., PDFs) for retrieval.
- **RAG Pipeline**: Retrieve relevant document chunks and generate answers using LLMs.
- **API Backend**: FastAPI server exposes endpoints for querying and managing the pipeline.
- **Frontend**: (Optional) Streamlit app for interactive querying and visualization.
- **Benchmarking**: Evaluate the performance and accuracy of the RAG pipeline.

### Typical Workflow
1. **Setup**: Install dependencies and configure environment variables.
2. **Start Backend**: Launch the FastAPI server to serve the RAG API.
3. **(Optional) Start Frontend**: Use the Streamlit app for a user-friendly interface.
4. **Query**: Send questions to the API or frontend; receive LLM-generated answers with supporting context.
5. **Benchmark**: Run benchmarking scripts to evaluate system performance.

## 📁 Project Structure
```
├── venv/                 # Virtual environment
├── main.py               # FastAPI application (backend)
├── rag_pipeline.py       # RAG processing logic
├── responder.py          # LLM response generation
├── utils.py              # Utility functions
├── streamlit_app.py      # Streamlit web interface (optional)
├── benchmarking.py       # Performance evaluation scripts
├── requirements.txt      # Python dependencies
├── activate_venv.bat     # Windows activation script
├── Datasets/             # Example datasets (e.g., policy.pdf)
├── metadata.pkl          # Metadata for document processing
├── README.md             # Project documentation
```

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <project-directory>
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```
**Or simply double-click:**
```
activate_venv.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the project root:
```
RAY_OPENAI_API_KEY=your_openai_api_key_here
```

## ▶️ Usage

### 1. Start the FastAPI Backend
```bash
python main.py
```
The API will be available at: [http://localhost:8000](http://localhost:8000)

### 2. (Optional) Run the Streamlit Frontend
```bash
streamlit run streamlit_app.py
```

### 3. (Optional) Run Benchmarking
```bash
python benchmarking.py
```

### 4. Deactivate the Virtual Environment
```bash
deactivate
```

## 🔑 Environment Variables
- `RAY_OPENAI_API_KEY`: Your OpenAI API key for LLM access. Required for backend operation.

## 🤝 Contributing
Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs before submitting a PR.

## 📄 License
Specify your license here (e.g., MIT, Apache 2.0, etc.)

---

**Contact:** For questions or support, please contact the project maintainer.
