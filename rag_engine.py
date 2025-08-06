# rag_engine.py
import numpy as np
import re
import PyPDF2
import io
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from collections import Counter
import string
import time

class EducationalRAG:
    def __init__(self, use_gemini_pro: bool = False, api_key: Optional[str] = None):
        """Initialize the RAG system optimized for educational use cases
        
        Args:
            use_gemini_pro: Whether to use Gemini Pro instead of local models
            api_key: Google API key (if None, will try to get from environment)
        """
        # EMBEDDING MODEL SELECTION (still using local model for embeddings)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # VECTOR DATABASE CONFIGURATION
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.metadata = []
        
        # QUESTION GENERATION MODEL SELECTION
        self.use_gemini_pro = use_gemini_pro
        
        if use_gemini_pro:
            # Configure Gemini Pro
            if api_key:
                genai.configure(api_key=api_key)
            else:
                # Try to get from environment variable
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                genai.configure(api_key=api_key)
            
            # Initialize Gemini Pro
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            print("Using Google Gemini Pro for question generation")
        else:
            # Use local model as fallback
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                
                self.qg_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
                self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "google/flan-t5-base",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.qg_pipeline = pipeline(
                    "text2text-generation",
                    model=self.qg_model,
                    tokenizer=self.qg_tokenizer,
                    max_new_tokens=300,
                    device=0 if torch.cuda.is_available() else -1
                )
                print("Using local FLAN-T5 model for question generation")
            except Exception as e:
                print(f"Warning: Could not load local model: {e}")
                print("Falling back to Gemini Pro (must be enabled with API key)")
                if not api_key:
                    raise ValueError("Must provide API key when local model fails")
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                self.use_gemini_pro = True
    
    def add_documents(self, documents: List[Dict[str, str]], chunk_size: int = 200) -> int:
        """
        Process and index educational documents with dynamic updates
        
        Args:
            documents: List of document dictionaries with title, source, and content
            chunk_size: Target size for text chunks in tokens
            
        Returns:
            Number of chunks added to the knowledge base
        """
        new_embeddings = []
        
        for doc in documents:
            # Text cleaning for educational content
            text = re.sub(r'\s+', ' ', doc['content']).strip()
            
            # Semantic chunking
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            char_count = 0
            
            for sent in sentences:
                if char_count + len(sent) > chunk_size * 4:  # Approx 1 token = 4 chars
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        self.chunks.append(chunk_text)
                        self.metadata.append({
                            'title': doc.get('title', 'Unnamed'),
                            'source': doc.get('source', 'Unknown'),
                            'chunk_id': len(self.chunks) - 1
                        })
                        new_embeddings.append(
                            self.embedding_model.encode([chunk_text])[0]
                        )
                        current_chunk = [sent]
                        char_count = len(sent)
                    else:
                        # Force chunk if single sentence exceeds limit
                        self.chunks.append(sent)
                        self.metadata.append({**doc, 'chunk_id': len(self.chunks)-1})
                        new_embeddings.append(
                            self.embedding_model.encode([sent])[0]
                        )
                else:
                    current_chunk.append(sent)
                    char_count += len(sent)
            
            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                self.chunks.append(chunk_text)
                self.metadata.append({
                    'title': doc.get('title', 'Unnamed'),
                    'source': doc.get('source', 'Unknown'),
                    'chunk_id': len(self.chunks) - 1
                })
                new_embeddings.append(
                    self.embedding_model.encode([chunk_text])[0]
                )
        
        # Add new embeddings to index
        if new_embeddings:
            new_embeddings = np.array(new_embeddings).astype('float32')
            self.index.add(new_embeddings)
        
        return len(new_embeddings)
    
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, Dict]]:
        """
        Retrieve relevant educational content based on semantic similarity
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        query_vec = self.embedding_model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                results.append((self.chunks[idx], self.metadata[idx]))
        return results
    
    def generate_questions(self, topic: str, num_questions: int = 3, 
                          question_types: List[str] = None) -> List[Dict]:
        """
        Generate pedagogically-sound questions using retrieved context
        
        Args:
            topic: The educational topic to generate questions about
            num_questions: Number of questions to generate
            question_types: List of question types to generate (e.g., ['mcq', 'short_answer'])
            
        Returns:
            List of question dictionaries with all relevant information
        """
        if question_types is None:
            question_types = ['short_answer', 'mcq']
            
        # Retrieve relevant educational content
        context_chunks = self.retrieve(topic, k=2)
        context = "\n\n".join([chunk for chunk, _ in context_chunks])
        
        questions = []
        difficulties = ['easy', 'medium', 'hard']
        
        for i in range(num_questions):
            difficulty = difficulties[i % 3]
            q_type = question_types[i % len(question_types)]
            
            # Generate question using appropriate model
            if self.use_gemini_pro:
                prompt = self._create_gemini_prompt(topic, context, difficulty, q_type)
                result = self._generate_with_gemini(prompt)
            else:
                if q_type == 'mcq':
                    prompt = self._create_mcq_prompt(topic, context, difficulty)
                else:  # short_answer
                    prompt = self._create_short_answer_prompt(topic, context, difficulty)
                
                result = self.qg_pipeline(
                    prompt, 
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True
                )[0]['generated_text']
            
            # Parse the result based on question type
            if q_type == 'mcq':
                question_data = self._parse_mcq(result, topic, difficulty, context_chunks)
            else:
                question_data = self._parse_short_answer(result, topic, difficulty, context_chunks)
            
            questions.append(question_data)
        
        return questions
    
    def _create_gemini_prompt(self, topic: str, context: str, difficulty: str, q_type: str) -> str:
        """Create a prompt optimized for Google Gemini Pro"""
        if q_type == 'mcq':
            return f"""You are an expert educator creating high-quality multiple-choice questions for students.
            
            Create ONE multiple-choice question about: {topic}
            
            Difficulty level: {difficulty}
            
            Educational context to base the question on:
            {context}
            
            INSTRUCTIONS:
            1. Create a clear, well-written multiple-choice question that tests understanding of key concepts
            2. Include 4 answer options (A, B, C, D) with exactly ONE correct answer
            3. Make the incorrect options plausible but clearly wrong to someone who understands the material
            4. The question should be answerable using ONLY the provided context
            5. Format your response EXACTLY as follows:
            
            QUESTION: [your question]
            A) [option 1]
            B) [option 2]
            C) [option 3]
            D) [option 4]
            ANSWER: [correct letter] [brief explanation of why it's correct]
            
            EXAMPLE:
            QUESTION: What is the primary function of an activation function in a neural network?
            A) To reduce the dimensionality of input data
            B) To introduce non-linearity into the model
            C) To normalize the input features
            D) To prevent overfitting during training
            ANSWER: B Activation functions introduce non-linearity, enabling neural networks to learn complex patterns. Without them, the network would only be capable of learning linear relationships."""
        
        else:  # short_answer
            return f"""You are an expert educator creating high-quality educational questions for students.
            
            Create ONE short answer question about: {topic}
            
            Difficulty level: {difficulty}
            
            Educational context to base the question on:
            {context}
            
            INSTRUCTIONS:
            1. Create a clear, well-written question that tests understanding of key concepts
            2. The question should require a concise but complete answer (2-4 sentences)
            3. The question should be answerable using ONLY the provided context
            4. Format your response EXACTLY as follows:
            
            QUESTION: [your question]
            ANSWER: [complete answer with educational explanation]
            
            EXAMPLE:
            QUESTION: What is supervised learning in machine learning?
            ANSWER: Supervised learning is a type of machine learning where algorithms are trained on labeled datasets. The algorithm learns patterns from input-output pairs, allowing it to predict outcomes for new, unseen data. Common applications include classification and regression tasks."""
    
    def _generate_with_gemini(self, prompt: str) -> str:
        """Generate text using Google Gemini Pro"""
        try:
            # Add retry logic for API calls
            for _ in range(3):
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=500
                        )
                    )
                    return response.text
                except Exception as e:
                    print(f"Error calling Gemini API: {e}")
                    time.sleep(2)  # Wait before retrying
            
            # Final attempt
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Failed to generate with Gemini: {e}")
            # Return fallback text
            return f"QUESTION: Explain key concepts about {topic}\nANSWER: Comprehensive explanation would appear here"
    
    def _create_mcq_prompt(self, topic: str, context: str, difficulty: str) -> str:
        """Create a prompt for generating multiple-choice questions (for local models)"""
        return f"""
Generate ONE multiple-choice question about: {topic}

Context for question generation:
{context}

Requirements:
1. Difficulty level: {difficulty}
2. Question must be answerable using ONLY the provided context
3. Include 4 answer options (A, B, C, D) with exactly ONE correct answer
4. Mark the correct answer with an asterisk (*) after the option letter
5. Include a brief explanation of why the correct answer is right
6. Format strictly as:

QUESTION: [your question]
A) [option 1]
B) [option 2]
C) [option 3]
D) [option 4]
ANSWER: [correct letter]* [explanation of why it's correct]

Example format:
QUESTION: What is the primary function of an activation function in a neural network?
A) To reduce the dimensionality of input data
B) To introduce non-linearity into the model*
C) To normalize the input features
D) To prevent overfitting during training
ANSWER: B* Activation functions introduce non-linearity, enabling neural networks to learn complex patterns. Without them, the network would only be capable of learning linear relationships.
"""
    
    def _create_short_answer_prompt(self, topic: str, context: str, difficulty: str) -> str:
        """Create a prompt for generating short answer questions (for local models)"""
        return f"""
Generate ONE educational question about: {topic}

Context for question generation:
{context}

Requirements:
1. Difficulty level: {difficulty}
2. Question must be answerable using ONLY the provided context
3. Include complete answer with explanation
4. Format strictly as:
   QUESTION: [your question]
   ANSWER: [answer with educational explanation]
"""
    
    def _parse_mcq(self, text: str, topic: str, difficulty: str, 
                  context_chunks: List[Tuple[str, Dict]]) -> Dict:
        """Parse the generated multiple-choice question text"""
        try:
            # Extract question
            question_match = re.search(r"QUESTION:\s*(.*?)(?=\n[A-D]\)|\nANSWER:)", text, re.DOTALL)
            if not question_match:
                # Try alternative pattern
                question_match = re.search(r"Question:\s*(.*?)(?=\n[A-D]\)|\nAnswer:)", text, re.DOTALL | re.IGNORECASE)
            
            if not question_match:
                raise ValueError("Could not find QUESTION section")
            
            question = question_match.group(1).strip()
            
            # Extract options
            options = {}
            for letter in ['A', 'B', 'C', 'D']:
                # Try standard format
                option_match = re.search(rf"{letter}[\\.)]\s*(.*?)(?=\n[A-D][\\.)]|\nANSWER:|$)", text, re.DOTALL)
                if not option_match:
                    # Try case-insensitive format
                    option_match = re.search(rf"{letter}[\\.)]\s*(.*?)(?=\n[A-D][\\.)]|\nAnswer:|$)", text, re.DOTALL | re.IGNORECASE)
                
                if option_match:
                    options[letter] = option_match.group(1).strip()
            
            if len(options) < 4:
                raise ValueError(f"Found only {len(options)} options, need 4")
            
            # Extract answer and explanation
            answer_match = re.search(r"ANSWER:\s*([A-D])\s+(.*)", text, re.DOTALL | re.IGNORECASE)
            if not answer_match:
                # Try alternative format
                answer_match = re.search(r"CORRECT:\s*([A-D])\s+(.*)", text, re.DOTALL | re.IGNORECASE)
                if not answer_match:
                    raise ValueError("Could not parse answer section")
            
            correct_letter = answer_match.group(1).upper()
            explanation = answer_match.group(2).strip()
            
            # Validate correct answer is in options
            if correct_letter not in options:
                # Try to find the correct answer by content
                for letter, option in options.items():
                    if explanation.lower() in option.lower() or option.lower() in explanation.lower():
                        correct_letter = letter
                        break
                else:
                    raise ValueError(f"Correct answer {correct_letter} not in options")
            
            return {
                'question_type': 'mcq',
                'question': question,
                'options': options,
                'correct_answer': correct_letter,
                'explanation': explanation,
                'difficulty': difficulty,
                'topic': topic,
                'context_source': [m['title'] for _, m in context_chunks]
            }
        except Exception as e:
            # Fallback in case parsing fails
            return {
                'question_type': 'mcq',
                'question': f"Multiple-choice question about {topic}",
                'options': {
                    'A': "Option A",
                    'B': "Option B",
                    'C': "Option C",
                    'D': "Option D"
                },
                'correct_answer': 'A',
                'explanation': "Explanation would appear here",
                'difficulty': difficulty,
                'topic': topic,
                'error': str(e),
                'context_source': [m['title'] for _, m in context_chunks]
            }
    
    def _parse_short_answer(self, text: str, topic: str, difficulty: str, 
                          context_chunks: List[Tuple[str, Dict]]) -> Dict:
        """Parse the generated short answer question text"""
        try:
            # Extract question
            question_match = re.search(r"QUESTION:(.*?)(?=ANSWER:)", text, re.DOTALL | re.IGNORECASE)
            if not question_match:
                question_match = re.search(r"Question:(.*?)(?=Answer:)", text, re.DOTALL | re.IGNORECASE)
            
            if not question_match:
                raise ValueError("Could not find QUESTION section")
            
            question = question_match.group(1).strip()
            
            # Extract answer
            answer_match = re.search(r"ANSWER:(.*)", text, re.DOTALL | re.IGNORECASE)
            if not answer_match:
                answer_match = re.search(r"Answer:(.*)", text, re.DOTALL | re.IGNORECASE)
            
            if not answer_match:
                raise ValueError("Could not find ANSWER section")
            
            answer = answer_match.group(1).strip()
            
            return {
                'question_type': 'short_answer',
                'question': question,
                'answer': answer,
                'difficulty': difficulty,
                'topic': topic,
                'context_source': [m['title'] for _, m in context_chunks]
            }
        except Exception as e:
            # Fallback in case parsing fails
            return {
                'question_type': 'short_answer',
                'question': f"Explain key concepts about {topic}",
                'answer': "Comprehensive explanation based on educational context",
                'difficulty': difficulty,
                'topic': topic,
                'error': str(e),
                'context_source': [m['title'] for _, m in context_chunks]
            }
    
    def get_knowledge_base_summary(self) -> Dict:
        """Provide system status for educational transparency"""
        if not self.chunks:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_dimension": self.dimension,
                "index_type": type(self.index).__name__,
                "source_distribution": {},
                "topic_distribution": {},
                "model_type": "Gemini Pro" if self.use_gemini_pro else "Local Model"
            }
        
        # Analyze document sources
        sources = [m['source'] for m in self.metadata]
        source_counts = Counter(sources)
        
        # Analyze topics (more generic approach for any subject)
        topic_keywords = {
            "math": ["math", "algebra", "calculus", "geometry", "equation", "theorem"],
            "science": ["science", "physics", "chemistry", "biology", "experiment", "hypothesis"],
            "history": ["history", "century", "war", "revolution", "empire", "ancient"],
            "literature": ["literature", "novel", "poem", "author", "character", "theme"],
            "computer science": ["computer", "algorithm", "programming", "code", "software", "data"],
            "general": ["concept", "theory", "principle", "idea", "understand", "learn"]
        }
        
        topics = []
        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            found_topic = False
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in chunk_lower for keyword in keywords):
                    topics.append(topic)
                    found_topic = True
                    break
            
            if not found_topic:
                topics.append("other")
        
        topic_counts = Counter(topics)
        
        return {
            "total_documents": len(set(m['title'] for m in self.metadata)),
            "total_chunks": len(self.chunks),
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "source_distribution": dict(source_counts),
            "topic_distribution": dict(topic_counts),
            "model_type": "Gemini Pro" if self.use_gemini_pro else "Local Model (FLAN-T5)"
        }

def extract_text_from_pdf(file) -> str:
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text