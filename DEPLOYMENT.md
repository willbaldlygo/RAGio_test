# Deployment Guide for HuggingFace Spaces

## Prerequisites

1. **HuggingFace Account**: Create an account at https://huggingface.co
2. **OpenAI API Key**: Get your API key from https://platform.openai.com
3. **GitHub Repository**: Your code should be in a GitHub repository

## Step-by-Step Deployment

### 1. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **Name**: `ragio-educational` (or your preferred name)
   - **License**: Apache 2.0
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free tier)
   - **Visibility**: Public

### 2. Connect Your Repository

1. In your new Space, go to "Settings"
2. Under "Repository", connect your GitHub repository
3. Set the branch to `main` or `master`

### 3. Add API Keys Securely

1. In your Space settings, go to "Repository secrets"
2. Add the following secrets:
   - **Key**: `OPENAI_API_KEY`, **Value**: Your OpenAI API key
   - **Key**: `HF_TOKEN`, **Value**: Your HuggingFace token (optional)

### 4. Configure Environment Variables

Add these in "Repository secrets" or "Variables":
- `DAILY_OPENAI_LIMIT`: 40
- `HOURLY_OPENAI_LIMIT`: 5
- `EMB_MODEL`: sentence-transformers/all-mpnet-base-v2
- `RERANK_MODEL`: cross-encoder/ms-marco-MiniLM-L-6-v2
- `HF_MODEL`: mistralai/Mistral-7B-Instruct-v0.2
- `OPENAI_MODEL`: gpt-3.5-turbo

### 5. Deploy

1. Push your code to GitHub
2. Your Space will automatically build and deploy
3. Monitor the build logs for any issues

## Security Best Practices

### ✅ DO:
- Use HuggingFace Secrets for API keys
- Set reasonable rate limits
- Monitor usage regularly
- Use environment variables for configuration

### ❌ DON'T:
- Commit API keys to your repository
- Set unlimited rate limits
- Ignore usage monitoring
- Use production API keys for educational spaces

## Cost Management

### Expected Costs (OpenAI):
- **GPT-3.5-turbo**: ~$0.002 per request
- **With 40 req/day limit**: ~$0.08 per user per day
- **50 students**: ~$4 per day maximum
- **Monthly estimate**: ~$120 maximum

### Cost Reduction Tips:
1. Default to HuggingFace models (free)
2. Set conservative rate limits
3. Monitor usage dashboard
4. Educate students about costs

## Monitoring and Maintenance

### Daily Checks:
- Monitor Space status
- Check error logs
- Review usage statistics

### Weekly Tasks:
- Analyze usage patterns
- Adjust rate limits if needed
- Update documentation

### Monthly Reviews:
- Review OpenAI billing
- Update dependencies
- Gather student feedback

## Troubleshooting

### Common Issues:

**Space won't start:**
- Check requirements.txt
- Verify all dependencies are available
- Check build logs for errors

**API errors:**
- Verify API keys in secrets
- Check rate limits
- Monitor quota usage

**Slow performance:**
- Consider upgrading to CPU upgrade (paid)
- Optimize chunk sizes
- Reduce batch sizes

### Getting Help:
- HuggingFace Discord: https://hf.co/join/discord
- HuggingFace Forums: https://discuss.huggingface.co
- GitHub Issues: Create issues in your repository

## Educational Setup Tips

### For Classroom Use:
1. Create a shared Space for the whole class
2. Set appropriate rate limits per student
3. Provide sample documents
4. Create guided exercises

### For Individual Learning:
1. Students can fork and create their own Spaces
2. Provide their own API keys
3. Experiment freely without limits

## Post-Deployment

### Share with Students:
1. Provide the Space URL
2. Create usage guidelines
3. Provide example prompts and documents
4. Set up feedback collection

### Monitor Success:
- Track student engagement
- Collect feedback on prompt experiments
- Monitor learning outcomes
- Iterate based on usage patterns

---

**Your Space URL will be**: `https://huggingface.co/spaces/YOUR_USERNAME/ragio-educational`

Good luck with your educational RAG deployment! 🚀
