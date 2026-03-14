import os
import sys

# Add src to Python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import gradio as gr
from src.pipeline import RAGPipeline

# Initialize RAG Pipeline globally so it's only loaded once when the app starts
print("Initializing RAG System...")
rag = RAGPipeline()
print("RAG System Ready!")

def ask_question(query):
    """Function to process the query and format output for Gradio UI"""
    if not query.strip():
        return "Please enter a question.", "No sources used."
        
    try:
        # Run pipeline
        result = rag.query(query)
        
        # Format answer
        answer = result.get("answer", "No answer generated.")
        
        # Format sources as a bulleted list with markdown links
        sources_list = result.get("sources", [])
        if sources_list:
            sources_md = "### Sources Used:\n"
            for src in sources_list:
                sources_md += f"- **{src.get('source', 'Unknown')}** ({src.get('section', '')}): [Link]({src.get('url', '#')})\n"
        else:
            sources_md = "No external context was retrieved for this answer."
            
        return answer, sources_md
        
    except Exception as e:
        return f"An error occurred: {str(e)}", "Error evaluating query."

# Build Gradio UI
with gr.Blocks(title="AI/ML RAG System Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 AI/ML Knowledge Base — RAG System")
    gr.Markdown("Ask questions domain-specific to Artificial Intelligence, Machine Learning, Deep Learning, and more. Backed by verified Wikipedia articles.")
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., What is a transformer model?",
                lines=2
            )
            submit_btn = gr.Button("Ask", variant="primary")
            
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="Answer", 
                lines=6,
                interactive=False
            )
            sources_output = gr.Markdown("Sources will appear here.")
            
    # Connect UI events
    submit_btn.click(
        fn=ask_question,
        inputs=[query_input],
        outputs=[answer_output, sources_output]
    )
    
    # Allow pressing "Enter" to submit
    query_input.submit(
        fn=ask_question,
        inputs=[query_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    # Check if vectorstore exists, if not, warn user
    if not os.path.exists("./vectorstore/chroma_db"):
        print("Warning: ChromaDB index not found at ./vectorstore/chroma_db.")
        print("Please run `python src/ingest.py` and `python src/embed.py` first.")
        
    print("Launching Gradio UI...")
    # Setting server_name and server_port for consistent demo experience
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
