# Domain-Specific Medical LLM Q&A Agent (RAG + LLM)

**A Production-Ready Retrieval-Augmented Generation System for Safe, Evidence-Based Medical Question Answering**

---

## 📋 Project Description

This project implements a **Domain-Specific Medical Q&A Agent** using **Retrieval-Augmented Generation (RAG)** combined with a powerful **LLM**. The system is designed to answer medical questions using **only trusted, publicly available medical guidelines** (e.g., WHO, CDC), making it suitable for health-tech applications.

The notebook is implemented in a **clean, modular, app-ready architecture**, making it easy to convert into a full production system.

---

## 🎯 Key Features

- ✅ **RAG Architecture**: Combines retrieval with LLM generation for accurate, evidence-based answers
- ✅ **Medical Domain Focus**: Uses only trusted medical guidelines (WHO, CDC)
- ✅ **Safety First**: Built-in safety checks, query validation, and medical disclaimers
- ✅ **Citation Support**: All answers include source citations for transparency
- ✅ **Production-Ready**: Modular, well-documented, and deployment-ready code
- ✅ **Comprehensive Evaluation**: Built-in evaluation metrics and benchmarking tools
- ✅ **Google Colab Optimized**: Ready to run in Google Colab with GPU support

---

## 🏗️ Architecture Overview

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances LLM responses by:

1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the LLM prompt with retrieved context
3. **Generating** answers based on the provided context

### Why RAG for Medical Applications?

- **Reduces Hallucinations**: LLMs alone can generate plausible but incorrect medical information
- **Evidence-Based**: Answers are grounded in actual medical guidelines
- **Up-to-Date**: Knowledge base can be updated without retraining the model
- **Transparency**: Citations show where information comes from
- **Safety**: Built-in validation prevents harmful responses

### Pipeline Flow:

```
Medical Documents → Chunking → Embeddings → Vector Store
                                                     ↓
User Query → Embedding → Vector Search → Retrieve Top-K Docs
                                                     ↓
Retrieved Context + Query → LLM Prompt → Generated Answer + Citations
```

---

## 📁 Project Structure

```
med_notebook/
├── Medical_RAG_QA_Agent.ipynb          # Main notebook file
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── .gitignore                           # Git ignore patterns
├── config/                              # Configuration management
│   └── config.yaml                     # Centralized configuration
├── data/                                # Sample data directory
│   ├── sample_medical_guidelines.txt   # Example medical text
│   └── test_questions.json             # Evaluation test suite
├── utils/                               # Modular utility modules
│   ├── __init__.py
│   ├── data_loader.py                  # Data ingestion with validation
│   ├── chunking.py                     # Advanced chunking strategies
│   ├── embeddings.py                   # Embedding generation & caching
│   ├── vector_store.py                 # FAISS wrapper with persistence
│   ├── retrieval.py                    # Retrieval with reranking
│   ├── rag_pipeline.py                 # End-to-end RAG pipeline
│   ├── evaluation.py                   # Evaluation metrics & benchmarking
│   └── safety.py                        # Medical domain safety checks
└── tests/                               # Unit tests (optional)
    └── test_rag_pipeline.py
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Open in Colab**: Upload the notebook to Google Colab
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)
3. **Run All Cells**: The notebook will install dependencies and run the complete pipeline
4. **Start Asking Questions**: Use the interactive Q&A section at the end

### Option 2: Local Installation

#### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

#### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd med_notebook
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook Medical_RAG_QA_Agent.ipynb
   ```

5. **Run all cells**: The notebook will guide you through the complete pipeline

---

## 📖 Usage

### Basic Usage

```python
from utils import RAGPipeline, RetrievalEngine, EmbeddingGenerator, VectorStore

# Initialize components (see notebook for full setup)
rag_pipeline = RAGPipeline(retrieval_engine, llm_model_name="google/flan-t5-base")

# Ask a question
result = rag_pipeline.generate_answer("What are the symptoms of malaria?")

# Get answer with citations
print(result['answer'])
print(f"Sources: {result['sources']}")
```

### Interactive Q&A

The notebook includes an interactive Q&A loop at the end. Simply run the cell and start asking medical questions!

---

## 🔧 Configuration

Configuration is managed through `config/config.yaml`. Key settings:

- **Embedding Model**: Choose your embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- **Chunking**: Adjust chunk size and overlap for your documents
- **Retrieval**: Configure top-k retrieval and similarity thresholds
- **LLM**: Select LLM model or configure API access

---

## 🧪 Evaluation

The project includes comprehensive evaluation tools:

- **Answer Quality Metrics**: Length, citation rate, source similarity
- **Benchmark Testing**: Test suite with medical questions
- **RAG vs Baseline Comparison**: Compare RAG with LLM-only responses

Run evaluation:
```python
from utils import RAGEvaluator

evaluator = RAGEvaluator(rag_pipeline)
results = evaluator.run_benchmark(test_questions)
```

---

## 🛡️ Safety Features

### Medical Domain Safety

- **Query Validation**: Detects emergency situations and personal advice requests
- **Harmful Content Detection**: Identifies potentially harmful queries
- **Medical Disclaimers**: Automatic disclaimers on all responses
- **Source Verification**: All answers cite trusted medical sources

### Safety Checks Include:

- Emergency situation detection (e.g., "chest pain", "difficulty breathing")
- Personal medical advice warnings
- Mental health crisis detection
- Appropriate medical disclaimers

---

## 📊 Technical Details

### Technologies Used

- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (CPU version for Colab compatibility)
- **LLM**: Hugging Face Transformers (`google/flan-t5-base` or configurable)
- **Chunking**: Custom medical document chunker with metadata extraction
- **Retrieval**: Cosine similarity search with optional reranking

### Performance Optimizations

- **Batch Processing**: Efficient embedding generation in batches
- **GPU Support**: Automatic GPU detection and utilization
- **Caching**: Embedding caching for faster repeated queries
- **Vector Store Persistence**: Save/load vector stores for faster startup

---

## 🚀 Deployment Options

The notebook structure can easily be converted into:

### 1. FastAPI Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_medical_question(request: QueryRequest):
    result = rag_pipeline.generate_answer(request.query)
    return result
```

### 2. Streamlit App

```python
import streamlit as st

st.title("Medical Q&A Assistant")
query = st.text_input("Ask a medical question:")
if query:
    result = rag_pipeline.generate_answer(query)
    st.write(result['answer'])
    st.write("Sources:", result['sources'])
```

### 3. Chatbot Interface

Integrate with popular chatbot frameworks (Discord, Slack, etc.)

---

## 📝 Adding Your Own Medical Data

1. **Prepare Documents**: Format your medical guidelines as text files
2. **Update Data Path**: Modify the data loading section in the notebook
3. **Re-run Pipeline**: The system will automatically process new documents

### Supported Formats:
- Plain text files (`.txt`)
- URLs (WHO, CDC websites)
- Structured JSON documents

---

## 🧪 Testing

The project includes a test suite with medical questions covering:
- Symptoms queries
- Treatment questions
- Prevention strategies
- Diagnosis information

Run tests:
```bash
python -m pytest tests/
```

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system provides general medical information for educational purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for personal medical concerns.

**Never use this system for:**
- Emergency medical situations
- Personal medical diagnosis
- Treatment decisions
- Medication recommendations

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional medical data sources
- Better embedding models fine-tuned on medical text
- Enhanced safety checks
- Multi-language support
- Improved evaluation metrics

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Yasin Ahmed Dema**

Created with ❤️ for safe, evidence-based medical information access.

---

## 🔗 Resources

- [WHO Guidelines](https://www.who.int/publications)
- [CDC Guidelines](https://www.cdc.gov/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

## 📧 Support

For issues, questions, or contributions, please open an issue on the repository.

---

## 🎯 Why This Project is Valuable

- **Demonstrates RAG**: Shows how retrieval-augmented generation works in practice
- **Medical Domain Expertise**: Addresses real-world healthcare information needs
- **Production-Ready**: Code quality suitable for deployment
- **Educational**: Well-documented for learning and teaching
- **Scalable**: Architecture supports adding more data and features

---

**Thank you for using the Medical RAG Q&A Agent!**

Remember: Always consult healthcare professionals for medical advice.

