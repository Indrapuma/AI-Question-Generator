# streamlit_app.py
import streamlit as st
from rag_engine import EducationalRAG, extract_text_from_pdf
import time
import matplotlib.pyplot as plt
from collections import Counter
import os

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'knowledge_base_built' not in st.session_state:
    st.session_state.knowledge_base_built = False
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'question_types' not in st.session_state:
    st.session_state.question_types = ['short_answer', 'mcq']
if 'use_gemini_pro' not in st.session_state:
    st.session_state.use_gemini_pro = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

def main():
    st.set_page_config(
        page_title="EduRAG - Educational Question Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    .topic-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .difficulty-easy {
        border-left: 4px solid #2ca02c;
    }
    .difficulty-medium {
        border-left: 4px solid #ff7f0e;
    }
    .difficulty-hard {
        border-left: 4px solid #d62728;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .header {
        color: #1f77b4;
        padding-bottom: 10px;
        border-bottom: 1px solid #e6e6e6;
        margin-bottom: 20px;
    }
    .mcq-option {
        margin: 8px 0;
        padding: 8px;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    .correct-answer {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .incorrect-answer {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .mcq-options {
        margin-top: 10px;
    }
    .model-badge {
        display: inline-block;
        background-color: #4a6fa5;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    .gemini-badge {
        background-color: #ea4335;
    }
    .local-model-badge {
        background-color: #4a6fa5;
    }
    .api-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.title("📚 EduRAG - Educational Question Generator")
    st.markdown("### Create personalized quizzes from your educational materials using AI")
    
    # Initialize RAG system if not already done
    if st.session_state.rag_system is None:
        # Check if Gemini Pro should be used
        use_gemini = st.session_state.use_gemini_pro
        api_key = st.session_state.api_key if use_gemini else None
        
        with st.spinner("Initializing AI models... This may take a moment"):
            try:
                st.session_state.rag_system = EducationalRAG(
                    use_gemini_pro=use_gemini,
                    api_key=api_key
                )
                st.session_state.model_initialized = True
            except Exception as e:
                st.session_state.model_initialized = False
                st.warning(f"Could not initialize model: {str(e)}")
    
    # Sidebar for knowledge base management
    with st.sidebar:
        st.header("📚 Knowledge Base")
        
        # Model selection
        st.subheader("AI Model Configuration")
        
        # Toggle for Gemini Pro
        use_gemini = st.checkbox("Use Google Gemini Pro", value=st.session_state.use_gemini_pro)
        
        if use_gemini != st.session_state.use_gemini_pro:
            st.session_state.use_gemini_pro = use_gemini
            st.session_state.rag_system = None  # Force reinitialization
            
            # Show warning if switching to Gemini without API key
            if use_gemini and not st.session_state.api_key:
                st.warning("Gemini Pro requires an API key. Please enter it below.")
            
            st.rerun()
        
        # API key input (only shown if using Gemini)
        if st.session_state.use_gemini_pro:
            api_key = st.text_input(
                "Google API Key", 
                value=st.session_state.api_key,
                type="password"
            )
            
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                st.session_state.rag_system = None  # Force reinitialization
                st.rerun()
            
            st.markdown("""
            <div class="api-warning">
            ⚠️ Your API key is stored only in your browser session and never sent to our servers.
            Get your API key from <a href="https://makersuite.google.com" target="_blank">Google AI Studio</a>
            </div>
            """, unsafe_allow_html=True)
        
        # Document upload section
        st.subheader("Add Educational Materials")
        uploaded_files = st.file_uploader(
            "Upload PDF or text documents", 
            type=["pdf", "txt"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            documents = []
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                        source = "PDF"
                    else:  # text/plain
                        text = file.getvalue().decode("utf-8")
                        source = "Text File"
                    
                    documents.append({
                        "title": file.name,
                        "source": source,
                        "content": text
                    })
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            if documents and st.button("Add to Knowledge Base", key="add_docs"):
                with st.spinner("Processing documents..."):
                    num_chunks = st.session_state.rag_system.add_documents(documents)
                    st.session_state.knowledge_base_built = True
                    st.success(f"Added {len(documents)} documents ({num_chunks} text chunks)")
        
        # Knowledge base summary
        st.subheader("Knowledge Base Status")
        if st.session_state.knowledge_base_built and st.session_state.rag_system:
            summary = st.session_state.rag_system.get_knowledge_base_summary()
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", summary["total_documents"])
            with col2:
                st.metric("Text Chunks", summary["total_chunks"])
            
            # Model information
            model_type = "gemini-2.0-flash" if st.session_state.use_gemini_pro else "Local Model"
            st.markdown(f"**AI Model:** {model_type}")
            
            # Source distribution
            if summary["source_distribution"]:
                st.markdown("**Source Distribution**")
                fig, ax = plt.subplots(figsize=(5, 3))
                sources = list(summary["source_distribution"].keys())
                counts = list(summary["source_distribution"].values())
                ax.pie(counts, labels=sources, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            # Topic distribution
            if summary["topic_distribution"]:
                st.markdown("**Topic Distribution**")
                fig, ax = plt.subplots(figsize=(5, 3))
                topics = list(summary["topic_distribution"].keys())
                counts = list(summary["topic_distribution"].values())
                ax.bar(topics, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
                plt.xticks(rotation=15)
                st.pyplot(fig)
        else:
            st.info("No documents added yet. Upload PDFs or text files to build your knowledge base.")
        
        st.markdown("---")
        st.subheader("Question Settings")
        st.session_state.question_types = st.multiselect(
            "Question types to generate",
            ["short_answer", "mcq"],
            default=st.session_state.question_types
        )
        
        st.markdown("### How It Works")
        st.markdown("""
        1. **Upload** educational materials (PDFs/text)
        2. **Build** knowledge base with AI indexing
        3. **Generate** questions on any topic
        4. **Review** AI-generated questions with answers
        """)
        
        st.markdown("### System Details")
        st.markdown(f"""
        - **Embedding Model**: all-MiniLM-L6-v2
        - **Vector Dimension**: 384
        - **Question Generator**: {"gemini-2.0-flash" if st.session_state.use_gemini_pro else "FLAN-T5 (Local)"}
        - **Search Method**: Semantic similarity
        """)

    # Main content area
    st.markdown('<div class="header">Generate Educational Questions</div>', unsafe_allow_html=True)
    
    # Model status indicator
    if st.session_state.use_gemini_pro:
        if st.session_state.api_key:
            st.markdown('<span class="model-badge gemini-badge">Using Google Gemini Pro</span>', 
                       unsafe_allow_html=True)
        else:
            st.warning("Google API key is required to use Gemini Pro. Please enter it in the sidebar.")
    else:
        st.markdown('<span class="model-badge local-model-badge">Using Local AI Model</span>', 
                   unsafe_allow_html=True)
    
    # Topic input section
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        topic = st.text_input(
            "Enter a topic for question generation", 
            value=st.session_state.current_topic,
            placeholder="e.g., Calculus, World War II, Shakespeare, Machine Learning"
        )
    with col2:
        num_questions = st.slider("Number of questions", 1, 5, 3)
    with col3:
        difficulty_filter = st.selectbox("Difficulty", ["All", "Easy", "Medium", "Hard"])
    
    if st.button("Generate Questions", disabled=not st.session_state.knowledge_base_built or not topic):
        st.session_state.processing = True
        st.session_state.current_topic = topic
        st.session_state.generated_questions = []
        
        # Check if we have what we need
        if st.session_state.use_gemini_pro and not st.session_state.api_key:
            st.error("Google API key is required to use Gemini Pro. Please enter it in the sidebar.")
            st.session_state.processing = False
            return
        
        with st.spinner(f"Generating {num_questions} questions about '{topic}'..."):
            try:
                questions = st.session_state.rag_system.generate_questions(
                    topic, 
                    num_questions,
                    st.session_state.question_types
                )
                st.session_state.generated_questions = questions
                st.session_state.processing = False
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.session_state.processing = False
    
    # Display generated questions
    if st.session_state.generated_questions:
        # Filter questions by difficulty if needed
        filtered_questions = st.session_state.generated_questions
        if difficulty_filter != "All":
            difficulty_map = {"Easy": "easy", "Medium": "medium", "Hard": "hard"}
            filtered_questions = [
                q for q in st.session_state.generated_questions 
                if q['difficulty'] == difficulty_map[difficulty_filter]
            ]
        
        st.markdown(f"## Questions on: *{st.session_state.current_topic}*")
        
        if not filtered_questions:
            st.warning("No questions match the selected difficulty level.")
        else:
            for i, q in enumerate(filtered_questions, 1):
                # Determine difficulty class
                difficulty_class = f"difficulty-{q['difficulty']}"
                
                with st.container():
                    st.markdown(f'<div class="topic-card {difficulty_class}">', unsafe_allow_html=True)
                    
                    # Question type badge
                    question_type_badge = {
                        'mcq': '🎯 Multiple Choice',
                        'short_answer': '✏️ Short Answer'
                    }.get(q['question_type'], '❓ Question')
                    
                    # Model badge
                    model_badge = "Google Gemini Pro" if st.session_state.use_gemini_pro else "Local AI Model"
                    
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="background-color: #4a6fa5; color: white; 
                              padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-right: 10px;">
                            {question_type_badge}
                        </span>
                        <span style="background-color: {'#2ca02c' if q['difficulty'] == 'easy' else '#ff7f0e' if q['difficulty'] == 'medium' else '#d62728'}; color: white; 
                              padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-right: 10px;">
                            {q['difficulty'].upper()} Difficulty
                        </span>
                        <span style="background-color: {'#ea4335' if st.session_state.use_gemini_pro else '#4a6fa5'}; color: white; 
                              padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-right: 10px;">
                            {model_badge}
                        </span>
                        <span style="margin-left: 10px; color: #666; font-style: italic;">
                            Source: {', '.join(q['context_source'])}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"### Q{i}: {q['question']}")
                    
                    # Display question type specific content
                    if q['question_type'] == 'mcq':
                        # Display MCQ options
                        st.markdown('<div class="mcq-options">', unsafe_allow_html=True)
                        for letter, option in q['options'].items():
                            if letter == q['correct_answer']:
                                st.markdown(f'<div class="mcq-option correct-answer"><strong>{letter})</strong> {option}</div>', 
                                           unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="mcq-option incorrect-answer"><strong>{letter})</strong> {option}</div>', 
                                           unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Explanation
                        with st.expander("View Explanation"):
                            st.markdown(f"**Correct Answer:** {q['correct_answer']}")
                            st.markdown(f"**Explanation:** {q['explanation']}")
                    
                    else:  # short_answer
                        with st.expander("View Answer and Explanation"):
                            st.markdown(f"**Answer:** {q['answer']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Empty state
    elif not st.session_state.knowledge_base_built:
        st.info("👈 Start by adding educational materials in the sidebar")
    elif not st.session_state.current_topic:
        st.info("✏️ Enter a topic above to generate questions")
    elif st.session_state.processing:
        st.info("⏳ Generating questions... This may take 15-30 seconds")
    else:
        st.info("❓ No questions generated yet. Click 'Generate Questions' to create your quiz!")

if __name__ == "__main__":
    main()