from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import traceback
import PyPDF2
import docx
from io import BytesIO
import re
from collections import Counter
import string
import difflib
import random

# Google Cloud imports
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from vertexai.language_models import TextGenerationModel
import google.auth

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS Configuration
CORS(app, 
     origins="*",
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
     supports_credentials=False)

# File processing limits
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_DOC_CHARS = 50000
MAX_TEXT_LENGTH = 100000
MIN_TEXT_LENGTH = 10

# Supported file types
SUPPORTED_MIME_TYPES = {
    'application/pdf': 'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/msword': 'doc',
    'text/plain': 'txt'
}

EXTENSION_TO_MIME = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.txt': 'text/plain'
}

# Google Cloud Configuration
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('VERTEX_LOCATION', 'us-central1')
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash-lite')

# In-memory storage for chat sessions
chat_sessions = {}
document_store = {}

# Initialize Vertex AI
def initialize_vertex_ai():
    """Initialize Vertex AI with credentials"""
    try:
        # Set up authentication
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # Test the connection
        model = GenerativeModel(MODEL_NAME)
        test_response = model.generate_content("Hello, test connection.")
        
        logger.info(f"✓ Vertex AI initialized successfully with model: {MODEL_NAME}")
        logger.info(f"✓ Project: {PROJECT_ID}, Location: {LOCATION}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize Vertex AI: {str(e)}")
        return False

def detect_mime_type(file_content: bytes, filename: str) -> str:
    """Detect MIME type using file signatures and extensions"""
    if file_content.startswith(b'%PDF'):
        return 'application/pdf'
    elif file_content.startswith(b'PK\x03\x04') or file_content.startswith(b'PK\x05\x06') or file_content.startswith(b'PK\x07\x08'):
        if filename and filename.lower().endswith('.docx'):
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif file_content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
        if filename and filename.lower().endswith('.doc'):
            return 'application/msword'
    
    if filename:
        _, ext = os.path.splitext(filename.lower())
        if ext in EXTENSION_TO_MIME:
            return EXTENSION_TO_MIME[ext]
    
    raise ValueError(f"Could not determine MIME type for file: {filename}")

def extract_text_fallback(file_content: bytes, mime_type: str) -> str:
    """Extract text using local libraries"""
    text = ""
    
    try:
        if mime_type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    text += f"\n--- Page {page_num + 1} (extraction failed) ---\n"
                
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = docx.Document(BytesIO(file_content))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        elif mime_type == 'text/plain':
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = file_content.decode('latin-1')
                except Exception as e:
                    raise Exception(f"Failed to decode text file: {e}")
        
        if not text.strip():
            raise Exception("No text could be extracted from the document")
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Text extraction failed: {e}")

class ChatSession:
    def __init__(self, session_id: str, document_id: str, document_title: str):
        self.session_id = session_id
        self.document_id = document_id
        self.document_title = document_title
        self.messages = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.context = {
            "conversation_history": []
        }
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Keep conversation history for context
        self.context["conversation_history"].append({
            'role': role,
            'content': content
        })
        
        # Keep only last 10 exchanges for context
        if len(self.context["conversation_history"]) > 20:
            self.context["conversation_history"] = self.context["conversation_history"][-20:]
            
        return message

class EnhancedLegalChatbot:
    """Enhanced AI-powered legal document chatbot that can handle both document-specific and general legal questions"""
    
    def __init__(self, document_text: str, document_title: str):
        self.document_text = document_text
        self.document_title = document_title
        self.model = GenerativeModel(MODEL_NAME)
        
        # Safety settings
        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            )
        ]
        
        # Generation config for better responses
        self.generation_config = {
            "max_output_tokens": 2048,
            "temperature": 0.3,
            "top_p": 0.8,
        }
    
    def classify_question_type(self, user_question: str) -> str:
        """Classify if the question is document-specific or general legal"""
        document_indicators = [
            'this document', 'the document', 'in this agreement', 'this contract',
            'this warrant', 'the agreement says', 'according to this',
            'what does this', 'in the document', 'this text', 'here it says'
        ]
        
        general_indicators = [
            'what is', 'what are', 'explain', 'define', 'how does', 'what does it mean when',
            'generally', 'typically', 'usually', 'in general', 'legal definition',
            'what happens if', 'how do', 'what are the types'
        ]
        
        question_lower = user_question.lower()
        
        # Check for explicit document references
        if any(indicator in question_lower for indicator in document_indicators):
            return "document_specific"
        
        # Check for general legal questions
        if any(indicator in question_lower for indicator in general_indicators):
            return "general_legal"
        
        # Default to hybrid approach for ambiguous questions
        return "hybrid"
    
    def create_document_specific_prompt(self) -> str:
        """Create a prompt for document-specific questions"""
        return f"""You are an expert legal document analyst. Your role is to help users understand and extract information from their legal documents.

DOCUMENT INFORMATION:
Title: {self.document_title}
Content: {self.document_text[:12000]}...

INSTRUCTIONS FOR DOCUMENT-SPECIFIC QUESTIONS:
1. Answer based ONLY on the information contained in this specific document
2. Quote relevant sections when possible
3. If the document doesn't contain the requested information, clearly state this
4. Be precise and accurate in your analysis
5. Reference specific clauses, sections, or terms mentioned in the document
6. Explain legal terms as they appear in the context of this document"""

    def create_general_legal_prompt(self) -> str:
        """Create a prompt for general legal questions"""
        return f"""You are an expert legal consultant with comprehensive knowledge of legal principles, terms, and concepts.

DOCUMENT CONTEXT (for reference):
The user has uploaded a document titled: {self.document_title}
Document type: Legal agreement/contract

INSTRUCTIONS FOR GENERAL LEGAL QUESTIONS:
1. Provide comprehensive explanations of legal concepts and terms
2. Give general legal knowledge and principles
3. Explain how legal concepts typically work in practice
4. Provide context and background information
5. When relevant, mention how the concept might relate to the type of document the user has
6. Include important disclaimers about seeking professional legal advice when appropriate
7. Be educational and informative

IMPORTANT: Always clarify whether you're providing general legal information vs. document-specific analysis."""

    def create_hybrid_prompt(self) -> str:
        """Create a prompt that can handle both document-specific and general questions"""
        return f"""You are an expert legal consultant and document analyst. You can provide both document-specific analysis and general legal knowledge.

DOCUMENT INFORMATION:
Title: {self.document_title}
Content: {self.document_text[:10000]}...

INSTRUCTIONS:
1. First, check if the question can be answered using the specific document content
2. If yes, provide document-specific information with quotes and references
3. Additionally, provide relevant general legal context and explanation
4. If the document doesn't contain specific information, provide general legal knowledge
5. Always clarify whether information comes from the document or general legal principles
6. Combine both approaches for comprehensive answers
7. Be educational while remaining accurate to the document content"""

    def generate_response(self, user_question: str, conversation_history: List[Dict] = None) -> str:
        """Generate AI response based on question type"""
        try:
            # Classify the question type
            question_type = self.classify_question_type(user_question)
            
            # Create context from conversation history
            context = ""
            if conversation_history and len(conversation_history) > 0:
                context = "\nRECENT CONVERSATION:\n"
                for msg in conversation_history[-6:]:  # Last 3 exchanges
                    context += f"{msg['role'].upper()}: {msg['content']}\n"
            
            # Choose appropriate prompt based on question type
            if question_type == "document_specific":
                system_prompt = self.create_document_specific_prompt()
                approach_note = "\n[APPROACH: Analyzing document content specifically]"
            elif question_type == "general_legal":
                system_prompt = self.create_general_legal_prompt()
                approach_note = "\n[APPROACH: Providing general legal information]"
            else:  # hybrid
                system_prompt = self.create_hybrid_prompt()
                approach_note = "\n[APPROACH: Combining document analysis with general legal knowledge]"
            
            # Create the complete prompt
            full_prompt = f"""{system_prompt}

{context}

USER QUESTION: {user_question}

Please provide a comprehensive and helpful answer. If this is about the document, be specific and quote relevant parts. If this is a general legal question, provide educational information. For ambiguous questions, provide both document-specific information (if available) and general legal context.

{approach_note}"""

            # Generate response using Vertex AI
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I'm having trouble generating a response right now. Could you please rephrase your question?"
                
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            return f"I encountered an issue processing your question. Please try rephrasing it or ask something else."
    
    def handle_greeting(self, message: str) -> Optional[str]:
        """Handle casual greetings"""
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        message_lower = message.lower().strip()
        
        if any(greeting in message_lower for greeting in greetings) and len(message.split()) <= 3:
            return f"""Hello! I'm your legal document assistant for '{self.document_title}'. 

I can help you with:
📄 **Document-specific questions** - Ask about specific clauses, terms, or content in your document
⚖️  **General legal questions** - Get explanations of legal concepts, terms, and principles
🔍 **Analysis and interpretation** - Understand what different parts of your document mean

What would you like to know?"""
        
        return None
    
    def handle_thanks(self, message: str) -> Optional[str]:
        """Handle thank you messages"""
        thanks = ['thanks', 'thank you', 'ty', 'thx']
        message_lower = message.lower().strip()
        
        if any(thank in message_lower for thank in thanks) and len(message.split()) <= 3:
            return "You're welcome! Feel free to ask me anything else about your document or any legal concepts you'd like to understand better."
        
        return None

def generate_intelligent_response(user_message: str, document_text: str, chat_history: List[Dict], document_title: str) -> str:
    """Main response generation using enhanced legal chatbot"""
    try:
        # Initialize the enhanced legal chatbot
        chatbot = EnhancedLegalChatbot(document_text, document_title)
        
        # Handle casual responses first
        greeting_response = chatbot.handle_greeting(user_message)
        if greeting_response:
            return greeting_response
            
        thanks_response = chatbot.handle_thanks(user_message)
        if thanks_response:
            return thanks_response
        
        # Get conversation history for context
        conversation_history = [
            msg for msg in chat_history 
            if msg.get('role') in ['user', 'assistant']
        ]
        
        # Generate AI response
        response = chatbot.generate_response(user_message, conversation_history)
        return response
        
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return "I'm having some technical difficulties right now. Could you please try asking your question again?"

# API Routes
@app.route('/upload-document', methods=['POST', 'OPTIONS'])
def upload_document():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        logger.info("=== DOCUMENT UPLOAD REQUEST ===")
        if 'document' not in request.files:
            return jsonify({'error': 'No document file provided'}), 400
        
        file = request.files['document']
        
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_content = file.read()
        
        if not file_content:
            return jsonify({'error': 'Empty file'}), 400
        
        file_size = len(file_content)
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large', 'message': f'File size exceeds {MAX_FILE_SIZE:,} bytes'}), 413
        
        logger.info(f"Processing file: {file.filename}, size: {file_size:,} bytes")
        
        try:
            mime_type = detect_mime_type(file_content, file.filename)
            logger.info(f"Detected MIME type: {mime_type}")
        except ValueError as e:
            return jsonify({'error': 'Unsupported file type', 'message': str(e)}), 400
        
        extracted_text = extract_text_fallback(file_content, mime_type)
        
        if len(extracted_text.strip()) < MIN_TEXT_LENGTH:
            return jsonify({'error': 'Extracted text too short', 'message': f'Text must be at least {MIN_TEXT_LENGTH} characters'}), 400
        
        document_id = str(uuid.uuid4())
        document_store[document_id] = {
            'id': document_id,
            'filename': file.filename,
            'text': extracted_text,
            'mime_type': mime_type,
            'file_size': file_size,
            'uploaded_at': datetime.now().isoformat(),
            'text_length': len(extracted_text)
        }
        
        session_id = str(uuid.uuid4())
        chat_session = ChatSession(session_id, document_id, file.filename)
        chat_sessions[session_id] = chat_session
        
        welcome_message = f"""🎉 Perfect! I've analyzed '{file.filename}' and I'm ready to be your comprehensive legal assistant.

I can help you with:
📋 **Document Analysis** - Ask about specific content, clauses, and terms in your document
⚖️ **Legal Explanations** - Get clear explanations of legal concepts and terminology  
🔍 **Both Combined** - Understand how general legal principles apply to your specific document

Try asking questions like:
• "What is a warrant agreement?" (general legal knowledge)
• "What does this document say about..." (document-specific)
• "Explain the terms mentioned in this agreement" (combined approach)

What would you like to explore first?"""

        chat_session.add_message('assistant', welcome_message)
        
        logger.info(f"Document uploaded successfully: {document_id}")
        
        response_data = {
            'status': 'success',
            'document_id': document_id,
            'session_id': session_id,
            'document_info': {
                'filename': file.filename,
                'file_size': file_size,
                'mime_type': mime_type,
                'text_length': len(extracted_text)
            },
            'welcome_message': welcome_message,
            'message': 'Document uploaded and analyzed successfully with Enhanced Legal AI!'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Upload failed', 'message': str(e)}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        chat_session = chat_sessions[session_id]
        
        if chat_session.document_id not in document_store:
            return jsonify({'error': 'Document not found'}), 404
        
        document = document_store[chat_session.document_id]
        
        # Add user message
        user_msg = chat_session.add_message('user', message)
        
        # Generate AI response
        ai_response = generate_intelligent_response(
            message, 
            document['text'], 
            chat_session.messages,
            document['filename']
        )
        
        # Add AI message
        ai_msg = chat_session.add_message('assistant', ai_response)
        
        logger.info(f"Enhanced chat response generated for session {session_id}")
        
        response_data = {
            'status': 'success',
            'user_message': user_msg,
            'ai_response': ai_msg,
            'session_info': {
                'session_id': session_id,
                'document_title': chat_session.document_title,
                'message_count': len(chat_session.messages)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Chat failed', 'message': str(e)}), 500

@app.route('/chat-history/<session_id>', methods=['GET', 'OPTIONS'])
def get_chat_history(session_id):
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        chat_session = chat_sessions[session_id]
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'document_title': chat_session.document_title,
            'messages': chat_session.messages,
            'created_at': chat_session.created_at.isoformat(),
            'last_activity': chat_session.last_activity.isoformat(),
            'message_count': len(chat_session.messages)
        })
        
    except Exception as e:
        logger.error(f"Get chat history failed: {str(e)}")
        return jsonify({'error': 'Failed to get chat history', 'message': str(e)}), 500

@app.route('/sessions', methods=['GET', 'OPTIONS'])
def get_sessions():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        sessions_list = []
        
        for session_id, session in chat_sessions.items():
            sessions_list.append({
                'session_id': session_id,
                'document_title': session.document_title,
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'message_count': len(session.messages)
            })
        
        sessions_list.sort(key=lambda x: x['last_activity'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'sessions': sessions_list,
            'total_sessions': len(sessions_list)
        })
        
    except Exception as e:
        logger.error(f"Get sessions failed: {str(e)}")
        return jsonify({'error': 'Failed to get sessions', 'message': str(e)}), 500

@app.route('/session/<session_id>', methods=['DELETE', 'OPTIONS'])
def delete_session(session_id):
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        del chat_sessions[session_id]
        
        return jsonify({
            'status': 'success',
            'message': 'Session deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Delete session failed: {str(e)}")
        return jsonify({'error': 'Failed to delete session', 'message': str(e)}), 500

@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        # Check Vertex AI status
        vertex_status = "connected" if initialize_vertex_ai() else "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'service': 'Enhanced Legal Document Chatbot',
            'version': '5.0.0',
            'ai_provider': 'Google Vertex AI',
            'model': MODEL_NAME,
            'vertex_ai_status': vertex_status,
            'capabilities': [
                'Document-specific analysis',
                'General legal knowledge',
                'Hybrid question handling',
                'Context-aware responses',
                'Legal term explanations'
            ],
            'statistics': {
                'active_sessions': len(chat_sessions),
                'documents_stored': len(document_store)
            },
            'features': [
                'Enhanced Legal Intelligence',
                'Dual-mode Question Handling',
                'Comprehensive Legal Analysis',
                'Context-Aware Responses',
                'Professional Legal Guidance'
            ]
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Enhanced Legal Document Chatbot',
        'version': '5.0.0',
        'description': 'Comprehensive legal assistant that handles both document-specific questions and general legal knowledge!',
        'ai_provider': 'Google Vertex AI',
        'model': MODEL_NAME,
        'capabilities': [
            '📄 Document-specific analysis and Q&A',
            '⚖️ General legal knowledge and explanations',
            '🔍 Hybrid approach for comprehensive answers',
            '💡 Legal term definitions and context',
            '🎯 Intelligent question classification'
        ],
        'example_questions': [
            'What is a warrant agreement? (general)',
            'What does this document say about transfer rights? (specific)',
            'Explain the key terms in this agreement (hybrid)'
        ]
    })

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    try:
        logger.info("=== STARTING ENHANCED LEGAL DOCUMENT CHATBOT ===")
        
        # Initialize Vertex AI
        if initialize_vertex_ai():
            logger.info("🤖 Google Vertex AI: ✓ Connected")
            logger.info(f"🧠 Model: {MODEL_NAME}")
            logger.info("⚖️ Legal Intelligence: ✓ Enhanced Mode")
            logger.info("📋 Document Analysis: ✓ Ready")
            logger.info("🔍 General Legal Knowledge: ✓ Enabled")
        else:
            logger.error("❌ Vertex AI initialization failed!")
            logger.error("Please check your Google Cloud credentials and configuration")
            exit(1)
            
        logger.info("Server starting on http://localhost:5001")
        
        app.run(debug=True, host='localhost', port=5001, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        exit(1)