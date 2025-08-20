import os
import logging
from jinja2 import Environment, FileSystemLoader
from time import perf_counter
import time
from collections import defaultdict

import gradio as gr

from backend.query_llm import generate_openai, generate_hf
from backend.semantic_search import retrieve
from backend.embedder import embedder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("TOP_K", 4))

env = Environment(loader=FileSystemLoader("./templates"))

template = env.get_template("template.j2")
template_html = env.get_template("template_html.j2")

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """Use the following unique documents in the Context section to answer the Query at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."""

# Rate limiting for OpenAI usage
class RateLimiter:
    def __init__(self):
        self.usage = defaultdict(list)  # user_id -> list of timestamps
        self.daily_limit = int(os.getenv("DAILY_OPENAI_LIMIT", 40))  # requests per day per user
        self.hourly_limit = int(os.getenv("HOURLY_OPENAI_LIMIT", 20))   # requests per hour per user
    
    def can_make_request(self, user_id: str) -> tuple[bool, str]:
        """Check if user can make a request. Returns (allowed, reason)"""
        now = time.time()
        user_requests = self.usage[user_id]
        
        # Clean old requests (older than 24 hours)
        user_requests[:] = [t for t in user_requests if now - t < 86400]
        
        # Check daily limit
        if len(user_requests) >= self.daily_limit:
            return False, f"Daily limit reached ({self.daily_limit} requests per day)"
        
        # Check hourly limit
        recent_requests = [t for t in user_requests if now - t < 3600]
        if len(recent_requests) >= self.hourly_limit:
            return False, f"Hourly limit reached ({self.hourly_limit} requests per hour)"
        
        return True, ""
    
    def record_request(self, user_id: str):
        """Record that user made a request"""
        self.usage[user_id].append(time.time())

rate_limiter = RateLimiter()

def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def bot(history, vs_name, api_kind, top_k: int, system_prompt: str, request: gr.Request):
    query = history[-1][0]

    if not query:
        raise gr.Warning("Please submit a non-empty string as a prompt")

    # Rate limiting for OpenAI
    if api_kind == "OpenAI":
        user_id = request.client.host if request else "unknown"
        can_request, reason = rate_limiter.can_make_request(user_id)
        if not can_request:
            raise gr.Error(f"Rate limit exceeded: {reason}")
        rate_limiter.record_request(user_id)

    logger.info("Retrieving documents...")

    doc_start = perf_counter()
    doc = retrieve(vs_name, query, k=25, rerank=True, top_k=top_k)
    doc_time = perf_counter() - doc_start
    logger.info(
        f"Finished Retrieving documents in \
        {round(doc_time, 2)} seconds..."
    )

    # Create Prompt with custom system prompt
    prompt = template.render(documents=doc, query=query, history=history, system_prompt=system_prompt)
    prompt_html = template_html.render(documents=doc, query=query, history=history, system_prompt=system_prompt)

    if api_kind == "HuggingFace":
        generate_fn = generate_hf
    elif api_kind == "OpenAI":
        generate_fn = generate_openai
    else:
        raise gr.Error(f"API {api_kind} is not supported")

    history[-1] = (history[-1][0], "")

    for character in generate_fn(prompt):
        history[-1] = (history[-1][0], character)
        yield history, prompt_html


def var_textbox(x):
    return x


def reset_system_prompt():
    return DEFAULT_SYSTEM_PROMPT


def get_usage_info(request: gr.Request):
    """Show current usage for this user"""
    if request:
        user_id = request.client.host
        user_requests = rate_limiter.usage[user_id]
        now = time.time()
        
        # Clean old requests
        user_requests[:] = [t for t in user_requests if now - t < 86400]
        
        recent_hour = len([t for t in user_requests if now - t < 3600])
        today = len(user_requests)
        
        return f"OpenAI Usage Today: {today}/{rate_limiter.daily_limit} | This Hour: {recent_hour}/{rate_limiter.hourly_limit}"
    return "Usage tracking unavailable"


with gr.Blocks(title="RAGio - Educational RAG with Custom System Prompts") as demo:

    vs_name_state = gr.State()

    # Educational header with warnings
    gr.Markdown("""
    # 📚 RAGio - Educational RAG System
    
    **⚠️ Educational Use Only**: This system uses a shared OpenAI API key with rate limits to prevent abuse.
    - OpenAI requests: 20 per hour, 40 per day per user
    - HuggingFace models: Unlimited (recommended for experimentation)
    
    **🎯 Learning Goal**: Understand how system prompts affect AI behavior in RAG systems.
    """)

    with gr.Row():
        file_input = gr.File(type="filepath", label="Upload PDF Document")
        upload_btn = gr.Button(value="Upload & Process", variant="primary")

    vs_name_output = gr.Textbox(label="Vector Store Name", interactive=False)

    upload_btn.click(embedder, inputs=file_input, outputs=vs_name_output).then(
        var_textbox, inputs=vs_name_output, outputs=vs_name_state
    )

    # System Prompt Section
    with gr.Group():
        gr.Markdown("## 🎯 System Prompt Configuration")
        gr.Markdown("**Educational Note:** The system prompt controls how the AI behaves. Try different instructions to see how it affects responses!")
        
        with gr.Row():
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=4,
                placeholder="Enter instructions for how the AI should behave...",
                info="This prompt tells the AI how to respond to user queries using the retrieved documents."
            )
        
        with gr.Row():
            reset_prompt_btn = gr.Button("Reset to Default", size="sm", variant="secondary")
            
        # Example prompts for educational purposes
        with gr.Accordion("📚 Example System Prompts to Try", open=False):
            gr.Markdown("""
            **Helpful Assistant:**
            ```
            You are a helpful assistant. Use the provided documents to answer questions thoroughly and helpfully. If you don't know something, say so clearly.
            ```
            
            **Concise Responder:**
            ```
            Provide brief, direct answers using only the information in the provided documents. Keep responses to 2-3 sentences maximum.
            ```
            
            **Critical Analyst:**
            ```
            Analyze the provided documents critically. Point out any limitations, biases, or gaps in the information before answering the question.
            ```
            
            **Creative Explainer:**
            ```
            Use the documents to answer questions, but explain concepts using analogies, examples, and creative explanations that make complex topics easy to understand.
            ```
            
            **Step-by-Step Tutor:**
            ```
            Act as a patient tutor. Break down complex information from the documents into step-by-step explanations. Always check if the student understands before moving on.
            ```
            """)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        avatar_images=(
            "https://aui.atlassian.com/aui/8.8/docs/images/avatar-person.svg",
            "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg",
        ),
        bubble_full_width=False,
        show_copy_button=True,
        show_share_button=True,
        height=400
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=3,
            show_label=False,
            placeholder="Ask a question about your uploaded document...",
            container=False,
        )
        txt_btn = gr.Button(value="Submit", scale=1, variant="primary")

    with gr.Row():
        api_kind_option = gr.Radio(
            choices=["HuggingFace", "OpenAI"], 
            value="HuggingFace", 
            label="API Provider",
            info="💡 Use HuggingFace for unlimited experimentation"
        )
        api_topk_slider = gr.Slider(
            minimum=1, 
            maximum=5, 
            value=3, 
            step=1, 
            label="Top-K Retrieved Documents",
            info="Number of most relevant document chunks to use"
        )

    # Usage tracking
    usage_display = gr.Textbox(label="API Usage", interactive=False)
    
    # Update usage display when page loads
    demo.load(get_usage_info, outputs=usage_display)

    # Reset system prompt functionality
    reset_prompt_btn.click(reset_system_prompt, outputs=system_prompt)

    prompt_html = gr.HTML(label="Current Prompt Structure")
    
    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot,
        [chatbot, vs_name_state, api_kind_option, api_topk_slider, system_prompt],
        [chatbot, prompt_html],
    ).then(
        get_usage_info, outputs=usage_display  # Update usage after request
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot,
        [chatbot, vs_name_state, api_kind_option, api_topk_slider, system_prompt],
        [chatbot, prompt_html],
    ).then(
        get_usage_info, outputs=usage_display  # Update usage after request
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)


if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True)
