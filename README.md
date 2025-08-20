---
title: RAGio Educational
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.31.4
app_file: app.py
pinned: false
suggested_hardware: cpu-basic
---

# RAGio - Educational RAG with Custom System Prompts

An interactive learning tool for understanding Retrieval-Augmented Generation (RAG) and prompt engineering. Perfect for students and educators exploring how different system prompts affect AI behavior.

## 🎓 Educational Features

- **Interactive System Prompt Editor**: Modify how the AI responds to queries
- **Real-time Prompt Visualization**: See exactly what prompt is sent to the model
- **Example Prompts**: Try different prompt styles to see their effects
- **Document Upload**: Upload your own PDFs to create custom knowledge bases
- **Multiple AI Providers**: Compare responses from HuggingFace and OpenAI models
- **Rate Limiting**: Learn about API constraints and cost management

## 🚀 How to Use

1. **Upload a Document**: Start by uploading a PDF file
2. **Customize System Prompt**: Edit the system prompt to change AI behavior
3. **Ask Questions**: Chat with your documents using the modified prompt
4. **Observe Changes**: See how different prompts affect responses
5. **Experiment**: Try the example prompts to understand different AI personalities

## 📚 Learning Objectives

Students will learn:
- How system prompts control AI behavior
- The impact of prompt engineering on response quality
- How RAG combines retrieval with generation
- The difference between various prompting strategies
- Real-world API constraints and rate limiting
- Cost considerations in AI applications

## 🔧 Technical Details

- **Vector Store**: LanceDB for efficient similarity search
- **Embeddings**: Sentence Transformers for document encoding
- **Reranking**: Cross-encoder for improved relevance
- **LLMs**: HuggingFace Transformers and OpenAI APIs
- **Security**: Rate limiting and API key protection

## 💡 Example System Prompts to Try

**Analytical**: "Analyze the provided documents critically and point out any limitations before answering."

**Creative**: "Explain concepts using analogies and examples to make them easy to understand."

**Concise**: "Provide brief, direct answers using only the document information. Maximum 2 sentences."

**Tutor**: "Act as a patient tutor. Break down complex information into step-by-step explanations."

## 🆓 Free Usage

This space runs on HuggingFace's free infrastructure using open-source models. OpenAI usage is rate-limited for educational purposes:
- **HuggingFace Models**: Unlimited usage (recommended for experimentation)
- **OpenAI Models**: 20 requests per hour, 40 per day per user

## 🔒 Security Features

- API keys stored securely in HuggingFace Secrets
- Per-user rate limiting to prevent abuse
- Usage tracking and monitoring
- Graceful fallback to free models

## Quick Start

Get started with RAGio by uploading a PDF and experimenting with different system prompts:

1. Upload any PDF document
2. Try the default system prompt
3. Modify the prompt using the examples provided
4. Ask questions and observe how responses change
5. Compare HuggingFace vs OpenAI model responses

## Contributing

Your contributions are welcome! If you have suggestions or want to improve RAGio, feel free to fork the repository, make changes, and submit a pull request.

---

*Built for educational purposes - experiment freely and learn about the fascinating world of RAG and prompt engineering!*
