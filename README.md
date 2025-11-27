# colab

A collection of Google Colab notebooks for various AI/ML projects.

---

## 📚 Notebooks in this Repository

### 1. Medical RAG Q&A Agent

**Domain-Specific Medical LLM Q&A Agent (RAG + LLM)**

A Production-Ready Retrieval-Augmented Generation System for Safe, Evidence-Based Medical Question Answering.

#### Key Features:
- ✅ **RAG Architecture**: Combines retrieval with LLM generation for accurate, evidence-based answers
- ✅ **Medical Domain Focus**: Uses only trusted medical guidelines (WHO, CDC)
- ✅ **Safety First**: Built-in safety checks, query validation, and medical disclaimers
- ✅ **Citation Support**: All answers include source citations for transparency
- ✅ **Production-Ready**: Modular, well-documented, and deployment-ready code
- ✅ **Google Colab Optimized**: Ready to run in Google Colab with GPU support

#### Quick Start:
1. Open `Medical_RAG_QA_Agent.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells
4. Start asking medical questions!

#### Project Structure:
```
med_notebook/
├── Medical_RAG_QA_Agent.ipynb    # Main notebook
├── utils/                         # Modular utility modules
├── config/                        # Configuration files
├── data/                          # Sample medical data
└── requirements.txt               # Dependencies
```

For detailed documentation, see the [Medical RAG Q&A Agent README](med_notebook/README.md).

---

### 2. Amharic Translation Notebook

Translation notebook for Amharic language processing.

File: `amh_translation.ipynb`

---

## 🚀 Getting Started

### Running Notebooks in Google Colab

1. **Upload to Colab**: 
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload the notebook file you want to run

2. **Enable GPU** (recommended for ML projects):
   - Runtime → Change runtime type → GPU

3. **Install Dependencies**:
   - Each notebook includes installation cells
   - Run the setup cells first

4. **Execute**:
   - Run all cells or execute step by step

---

## 📋 Requirements

Each notebook has its own `requirements.txt` file. Install dependencies as needed:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Feel free to add your own notebooks to this collection!

---

## ⚠️ Medical Disclaimer

**For Medical RAG Q&A Agent**: This system provides general medical information for educational purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for personal medical concerns.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Yasin Ahmed Dema**

---

## 🔗 Resources

- [Google Colab](https://colab.research.google.com/)
- [Hugging Face](https://huggingface.co/)
- [WHO Guidelines](https://www.who.int/publications)
- [CDC Guidelines](https://www.cdc.gov/)
