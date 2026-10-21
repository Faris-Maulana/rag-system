import os
import sys

# Add src to Python path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import gradio as gr  # noqa: E402

from config import CHROMA_PERSIST_DIR  # noqa: E402
from pipeline import RAGPipeline  # noqa: E402


def format_sources(sources_list):
    if not sources_list:
        return "No external context was retrieved for this answer."
    sources_md = "### Sources Used:\n"
    for src in sources_list:
        sources_md += f"- **{src.get('source', 'Unknown')}** ({src.get('section', '')}): [Link]({src.get('url', '#')})\n"
    return sources_md

def build_demo(rag):
    def ask_question(query):
        """Process the query and format output for the Gradio UI"""
        if not query.strip():
            return "Please enter a question.", "No sources used."
        try:
            result = rag.query(query)
            return result.get("answer", "No answer generated."), format_sources(result.get("sources", []))
        except Exception as e:
            return f"An error occurred: {str(e)}", "Error evaluating query."

    with gr.Blocks(title="AI/ML RAG System Demo") as demo:
        gr.Markdown("# AI/ML Knowledge Base — RAG System")
        gr.Markdown("Ask questions about Artificial Intelligence, Machine Learning and Deep Learning. Answers are grounded in 54 Wikipedia articles and cite their sources.")

        query_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What is a transformer model?",
            lines=2
        )
        submit_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Textbox(label="Answer", lines=6, interactive=False)
        sources_output = gr.Markdown("Sources will appear here.")

        submit_btn.click(fn=ask_question, inputs=[query_input], outputs=[answer_output, sources_output])
        query_input.submit(fn=ask_question, inputs=[query_input], outputs=[answer_output, sources_output])
    return demo

if __name__ == "__main__":
    if not os.path.exists(CHROMA_PERSIST_DIR):
        sys.exit(f"ChromaDB index not found at {CHROMA_PERSIST_DIR}. Run `make index` first.")

    print("Initializing RAG System...")
    try:
        rag = RAGPipeline()
    except ValueError as e:  # missing GROQ_API_KEY
        sys.exit(str(e))
    demo = build_demo(rag)
    print("Launching Gradio UI...")
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
