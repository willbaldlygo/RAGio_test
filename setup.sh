#!/bin/bash

# RAGio Quick Setup Script
echo "🚀 Setting up RAGio Educational..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Install requirements
echo "📦 Installing requirements..."
pip3 install -r requirements.txt

# Check if .env exists, if not copy from template
if [ ! -f ".env" ]; then
    echo "📝 Setting up environment file..."
    cp .template.env .env
    echo "⚠️  Please edit .env file with your API keys before running!"
    echo "   You need to add your OPENAI_API_KEY and HF_TOKEN"
else
    echo "✅ Environment file already exists"
fi

# Download spacy model
echo "🔤 Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source .env"
echo "3. Run: python3 app.py"
echo "4. Open http://127.0.0.1:7860 in your browser"
echo ""
echo "🎓 For deployment to HuggingFace Spaces, see DEPLOYMENT.md"
