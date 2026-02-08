"""
Advanced RAG Pipeline using LangChain
Components: Document loaders, smart chunking, hybrid retrieval, re-ranking, compression
"""

import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

COLLEGE_NAME = "Garden City University"

# LangChain Document Loaders
from langchain_community.document_loaders import (PyPDFLoader, CSVLoader,
                                        UnstructuredURLLoader)
from langchain_community.tools import DuckDuckGoSearchRun

# LangChain Text Splitters
from langchain.text_splitter import (RecursiveCharacterTextSplitter,
                                     CharacterTextSplitter)

# LangChain Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
# Uncomment for OpenAI:
# from langchain_openai import OpenAIEmbeddings

# LangChain Vector Stores
from langchain_community.vectorstores import Chroma
# Uncomment for Pinecone:
# from langchain_community.vectorstores import Pinecone

# LangChain Retrievers
from langchain.retrievers import (BM25Retriever, EnsembleRetriever)
try:
    # BaseRetriever location may vary across langchain versions
    from langchain.retrievers.base import BaseRetriever
except Exception:
    from langchain.schema import BaseRetriever
# pydantic PrivateAttr for storing non-field attributes on BaseRetriever
from pydantic import PrivateAttr

# LangChain Chains
from langchain.chains import (ConversationalRetrievalChain, LLMChain)
from langchain.chains.question_answering import load_qa_chain

# LangChain Memory
from langchain.memory import ConversationBufferMemory

# LangChain Prompts
from langchain.prompts import PromptTemplate

# LangChain LLMs
from langchain_google_genai import ChatGoogleGenerativeAI
# Fallback to HuggingFace:
from langchain_community.llms import HuggingFacePipeline

# Additional imports
from langchain.schema import Document

# Simple retriever wrapper used to call the underlying ensemble retriever
# and then optionally rerank the returned documents using the cross-encoder.
class SimpleRetriever(BaseRetriever):
    """A thin wrapper that implements BaseRetriever so LangChain chains
    accept it. It delegates retrieval to an underlying retriever and then
    applies manual reranking via the parent AdvancedRAGRetriever.rerank.

    Because BaseRetriever is a pydantic model, store non-field attributes as
    PrivateAttr to avoid pydantic validation errors when assigning them.
    """

    _base_retriever: any = PrivateAttr()
    _parent: any = PrivateAttr()
    _k: int = PrivateAttr()

    def __init__(self, base_retriever, parent: "AdvancedRAGRetriever", k: int = 10):
        # Initialize private attributes
        super().__init__()
        self._base_retriever = base_retriever
        self._parent = parent
        self._k = k

    def get_relevant_documents(self, query: str):
        """Synchronous retrieval entrypoint expected by many LangChain chains."""
        # Use the base retriever to get candidate documents
        try:
            docs = self._base_retriever.get_relevant_documents(query)
        except Exception:
            # Some retrievers use retrieve or __call__
            try:
                docs = self._base_retriever.retrieve(query)
            except Exception:
                try:
                    docs = self._base_retriever(query)
                except Exception:
                    docs = []

        # Rerank using parent's rerank method
        try:
            top_docs = self._parent.rerank(query, docs, top_k=self._k)
        except Exception:
            top_docs = docs[: self._k]

        return top_docs

    async def aget_relevant_documents(self, query: str):
        """Async wrapper - call sync method in thread if underlying retriever
        doesn't provide async version."""
        return self.get_relevant_documents(query)
import pandas as pd


def initialize_vector_store(collection_name: str = "college_chatbot",
                            persist_directory: str = "./chroma_db") -> Chroma:
    """Initialize ChromaDB vector store with embeddings"""

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})

    # Uncomment for OpenAI embeddings:
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize ChromaDB
    vector_store = Chroma(collection_name=collection_name,
                          embedding_function=embeddings,
                          persist_directory=persist_directory)

    return vector_store


class DocumentIngestionPipeline:
    """
    Document ingestion using LangChain loaders and smart chunking
    """

    def __init__(self,
                 vector_store: Chroma,
                 preloaded_csvs: Optional[Dict[str, pd.DataFrame]] = None,
                 preloaded_pdfs: Optional[Dict[str, List[Document]]] = None):
        self.vector_store = vector_store
        self.preloaded_csvs = preloaded_csvs or {}
        self.preloaded_pdfs = preloaded_pdfs or {}

        # Initialize text splitters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""])

        # Specialized splitter for tables/CSVs
        self.table_splitter = CharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=0,
                                                    separator="\n")

    def ingest_file(self, file_path: str, filename: str) -> List[Document]:
        """Ingest a file using appropriate LangChain loader"""

        if file_path.endswith('.pdf'):
            return self._ingest_pdf(file_path, filename)
        elif file_path.endswith('.csv'):
            return self._ingest_csv(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _ingest_pdf(self, file_path: str, filename: str) -> List[Document]:
        """Ingest PDF using PyPDFLoader with smart chunking"""

        # Load PDF with page awareness (prefer cached pages)
        if filename in self.preloaded_pdfs:
            pages = [
                Document(page_content=doc.page_content,
                         metadata=dict(doc.metadata))
                for doc in self.preloaded_pdfs[filename]
            ]
        else:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

        # Add custom metadata
        for page in pages:
            page.metadata['source'] = filename
            page.metadata['type'] = 'pdf'

            # Extract metadata from content
            self._enrich_metadata(page)

        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        return chunks

    def _ingest_csv(self, file_path: str, filename: str) -> List[Document]:
        """Ingest CSV using CSVLoader with structured data preservation"""

        # Load CSV
        if filename in self.preloaded_csvs:
            df = self.preloaded_csvs[filename].copy()
        else:
            df = pd.read_csv(file_path)

        # Clean column names
        df.columns = df.columns.str.strip()

        documents = []

        # Strategy 1: Each row as a document
        for idx, row in df.iterrows():
            # Format row as structured text
            content = self._format_csv_row(row, df.columns)

            metadata = {
                'source': filename,
                'type': 'csv',
                'row': idx + 1,
                'timestamp': datetime.now().isoformat()
            }

            # Extract metadata from row
            metadata.update(self._extract_csv_metadata(row))

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        # Strategy 2: Table summary document
        table_summary = f"Complete table from {filename}:\n\n"
        table_summary += f"Columns: {', '.join(df.columns)}\n"
        table_summary += f"Total rows: {len(df)}\n\n"
        table_summary += df.head(10).to_string(index=False)

        summary_doc = Document(page_content=table_summary,
                               metadata={
                                   'source': filename,
                                   'type': 'csv_summary',
                                   'rows': len(df)
                               })
        documents.append(summary_doc)

        # Add to vector store
        self.vector_store.add_documents(documents)

        return documents

    def ingest_url(self, url: str) -> List[Document]:
        """Ingest webpage using UnstructuredURLLoader"""

        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()

        # Add metadata
        for doc in documents:
            doc.metadata['source'] = url
            doc.metadata['type'] = 'webpage'
            self._enrich_metadata(doc)

        # Split and add to vector store
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)

        return chunks

    def _format_csv_row(self, row: pd.Series, columns: pd.Index) -> str:
        """Format CSV row as natural language"""
        parts = []
        for col, val in zip(columns, row):
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        return " | ".join(parts)

    def _extract_csv_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata from CSV row"""
        metadata = {}

        for col in row.index:
            col_lower = col.lower()
            value = str(row[col]) if pd.notna(row[col]) else ""

            if 'branch' in col_lower or 'course' in col_lower:
                metadata['branch'] = value
            elif 'specialization' in col_lower or 'spec' in col_lower:
                metadata['specialization'] = value
            elif 'fee' in col_lower or 'cost' in col_lower:
                metadata['contains_fee'] = True
            elif 'year' in col_lower or 'semester' in col_lower:
                metadata['year'] = value
            elif 'eligibility' in col_lower:
                metadata['eligibility'] = value

        return metadata

    def _enrich_metadata(self, doc: Document):
        """Extract and add metadata from document content"""
        content = doc.page_content

        # Extract branch/specialization
        branch_patterns = [
            r'(?i)\b(CSE|ECE|Mechanical|Civil|AI|ML|Data Science|Cybersecurity|IoT|Blockchain)\b',
            r'(?i)\b(B\.?Tech|M\.?Tech|MBA|MCA)\b'
        ]

        for pattern in branch_patterns:
            match = re.search(pattern, content)
            if match:
                doc.metadata['branch'] = match.group(1)
                break

        # Extract year/semester
        year_match = re.search(
            r'(?i)(first|second|third|fourth|1st|2nd|3rd|4th)\s+(year|semester)',
            content)
        if year_match:
            doc.metadata['academic_level'] = year_match.group(0)

        # Extract fee information
        if re.search(r'₹\s?[\d,]+|Rs\.?\s?[\d,]+', content):
            doc.metadata['contains_fee'] = True


class QueryRewriter:
    """Rewrite user queries using LLMChain for better retrieval"""

    def __init__(self):
        # Initialize LLM for query rewriting
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0,
                google_api_key=os.getenv('GEMINI_API_KEY'))
        except:
            # Fallback to simple rule-based rewriting
            self.llm = None

        # Query rewriting prompt
        self.rewrite_prompt = PromptTemplate(
            input_variables=["original_query", "conversation_history"],
            template=
            """You are a query rewriting assistant for a college information chatbot.

Rewrite the user's query to be more explicit and searchable. Consider:
1. Add missing context from conversation history
2. Expand abbreviations (CSE → Computer Science Engineering)
3. Make implicit information explicit (e.g., "fees" → "annual fee and total fee")
4. Add relevant keywords

Conversation History:
{conversation_history}

Original Query: {original_query}

Rewritten Query (output ONLY the rewritten query, no explanation):""")

        if self.llm:
            self.rewrite_chain = LLMChain(llm=self.llm,
                                          prompt=self.rewrite_prompt)

    def rewrite(self, query: str, conversation_history: str = "") -> str:
        """Rewrite query for better retrieval"""

        if self.llm:
            try:
                result = self.rewrite_chain.run(
                    original_query=query,
                    conversation_history=conversation_history)
                return result.strip()
            except:
                pass

        # Fallback: rule-based rewriting
        return self._rule_based_rewrite(query)

    def _rule_based_rewrite(self, query: str) -> str:
        """Simple rule-based query expansion"""
        expansions = {
            r'\bcse\b': 'CSE Computer Science Engineering',
            r'\bece\b': 'ECE Electronics Communication Engineering',
            r'\bfee\b': 'fee annual fee total fee cost',
            r'\bai\b': 'AI Artificial Intelligence Machine Learning',
            r'\bcybersecurity\b': 'cybersecurity information security',
            r'\blateral\b': 'lateral entry admission eligibility'
        }

        expanded = query
        for pattern, expansion in expansions.items():
            expanded = re.sub(pattern,
                              expansion,
                              expanded,
                              flags=re.IGNORECASE)

        return expanded


class AdvancedRAGRetriever:
    """
    Advanced retriever with:
    - Hybrid search (Vector + BM25)
    - Cross-encoder re-ranking
    - Context compression
    """

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.query_rewriter = QueryRewriter()

        # Build retrievers
        # We'll configure the cross-encoder re-ranker inside _build_retrievers
        self._build_retrievers()

    def _build_retrievers(self):
        """Build hybrid retriever with vector + BM25"""

        # Vector retriever
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10})

        # BM25 retriever (keyword-based)
        try:
            all_docs = self.vector_store.get()['documents']
            if all_docs:
                # Create documents for BM25
                docs_for_bm25 = [
                    Document(page_content=doc) for doc in all_docs
                ]
                self.bm25_retriever = BM25Retriever.from_documents(
                    docs_for_bm25)
                self.bm25_retriever.k = 10
            else:
                self.bm25_retriever = None
        except:
            self.bm25_retriever = None

        # Ensemble retriever (hybrid)
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]  # 70% vector, 30% BM25
            )
        else:
            self.ensemble_retriever = self.vector_retriever

        # We'll use a sentence-transformers CrossEncoder instance for manual
        # re-ranking to avoid Pydantic validation issues with LangChain's
        # CrossEncoderReranker wrapper.
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception:
            self.cross_encoder = None

        # Compression retriever - use a SimpleRetriever that wraps the
        # ensemble retriever and applies manual reranking
        self.compression_retriever = SimpleRetriever(self.ensemble_retriever,
                                                     parent=self,
                                                     k=10)

    def get_retriever(self):
        """Get the full retrieval pipeline (a retriever-like object)."""
        return self.compression_retriever

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Re-rank documents using the cross-encoder (if available).

        Returns the top_k documents sorted by relevance score.
        """
        if not getattr(self, 'cross_encoder', None) or not docs:
            return docs[:top_k]

        # Prepare pairs for cross-encoder scoring
        pairs = [(query, d.page_content) for d in docs]
        try:
            scores = self.cross_encoder.predict(pairs)
        except Exception:
            # If prediction fails, return original ordering
            return docs[:top_k]

        # Attach scores and sort
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = [d for d, s in scored[:top_k]]
        return top_docs

    def retrieve_with_rewriting(
            self,
            query: str,
            conversation_history: str = "") -> tuple[List[Document], str]:
        """Retrieve documents with query rewriting"""

        # Rewrite query
        rewritten_query = self.query_rewriter.rewrite(query,
                                                      conversation_history)

        # Retrieve documents
        docs = self.compression_retriever.get_relevant_documents(
            rewritten_query)

        return docs, rewritten_query


class ConversationalRAGChain:
    """
    Conversational RAG chain with memory using LangChain
    """

    def __init__(self, rag_retriever: AdvancedRAGRetriever):
        self.rag_retriever = rag_retriever
        self.sessions = {}  # Store memory per session

        # Initialize LLM
        try:
            # ChatGoogleGenerativeAI expects the API key under the
            # 'google_api_key' parameter name. Read the user's GEMINI_API_KEY
            # environment variable and pass it as google_api_key so the
            # client is initialized correctly.
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.3,
                google_api_key=os.getenv('GEMINI_API_KEY'))
        except:
            # Fallback to HuggingFace
            from transformers import pipeline
            hf_pipeline = pipeline("text2text-generation",
                                   model="google/flan-t5-base",
                                   max_length=512,
                                   device=-1)
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Create QA prompt
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=
            """You are a helpful college information assistant. Answer questions using ONLY the provided context.

Rules:
1. Be concise and direct
2. Always cite sources (document name and page number)
3. If information is not in the context, say: "I don't have information about that. Please contact the admissions office at admissions@college.edu"
4. For fees, include both annual and total amounts if available
5. Be specific about branches, specializations, and years

Context:
{context}

Question: {question}

Answer (include citations):""")

    def _get_or_create_memory(self,
                              session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for session"""

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'memory':
                ConversationBufferMemory(memory_key="chat_history",
                                         return_messages=True,
                                         output_key='answer'),
                'created_at':
                datetime.now(),
                'message_count':
                0
            }

        return self.sessions[session_id]['memory']

    def check_query_relevance(self, query: str) -> Dict[str, bool]:
        """
        Check if the query is relevant to the college domain using LLM.
        Returns a dict: {'is_relevant': bool, 'is_general_knowledge': bool}
        """
        if not self.llm:
            return {'is_relevant': True, 'is_general_knowledge': False}  # Default to allowing if no LLM

        try:
            # Simple prompt to classify the query
            # We use a zero-shot classification approach
            
            check_prompt = PromptTemplate(
                input_variables=["query", "college_name"],
                template="""You are a strict guard for a college chatbot for {college_name}.
Your job is to classify if a query is relevant to the college/education domain or if it is general knowledge (like "Who is PM of India?", "What is the capital of France?", "Current cricket score", etc.).

Query: {query}

Rules:
1. "Who is the PM?", "Who is CM?", "Weather in London", "Cricket score" -> NOT RELEVANT (General Knowledge)
2. "What are the fees?", "Courses offered", "Admission process", "Hostel facilities", "Location of college", "Placement records" -> RELEVANT
3. "Is there a gym?", "Library timings" -> RELEVANT (implies college gym/library)
4. "Tell me about the college" -> RELEVANT
5. Greetings like "Hi", "Hello" -> RELEVANT

Do not answer the query. Just output structured classification.

Output format (JSON):
{{"relevant": true/false, "general_knowledge": true/false}}

JSON Output:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=check_prompt)
            result = chain.run(query=query, college_name=COLLEGE_NAME)
            
            # clean result
            import json
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:-3]
            if result.startswith("```"):
                result = result[3:-3]
                
            data = json.loads(result)
            return {'is_relevant': data.get('relevant', True), 'is_general_knowledge': data.get('general_knowledge', False)}
            
        except Exception as e:
            print(f"Relevance check failed: {e}")
            return {'is_relevant': True, 'is_general_knowledge': False}

    def process_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """Process query through conversational RAG pipeline"""

        # Step 0: Check Relevance
        relevance = self.check_query_relevance(query)
        if not relevance['is_relevant'] and relevance['is_general_knowledge']:
            return {
                'answer': f"I am a specialized chatbot for {COLLEGE_NAME}. I cannot answer general knowledge questions like '{query}'. Please ask me about admissions, courses, fees, or campus life.",
                'source_documents': [],
                'rewritten_query': query
            }

        # Get memory
        memory = self._get_or_create_memory(session_id)

        # Get conversation history as string
        history_str = self._format_history(memory)

        # Retrieve with query rewriting
        docs, rewritten_query = self.rag_retriever.retrieve_with_rewriting(
            query, history_str)

        # Step 1: Try Local Answer
        # Even if docs are found, they might be irrelevant.
        # But for now, we assume if we found 'something', we try to answer.
        # The key is to catch the "I don't know" from the QA chain.
        
        answer_text = ""
        top_docs = []
        
        if docs:
            # Rerank retrieved docs (use cross-encoder if available)
            try:
                top_docs = self.rag_retriever.rerank(query, docs, top_k=5)
            except Exception:
                top_docs = docs[:5]

            # Build context from top documents (include simple citations)
            try:
                context_parts = []
                for d in top_docs:
                    src = d.metadata.get('source', 'unknown')
                    snippet = d.page_content
                    context_parts.append(f"Source: {src}\n{snippet}")

                context_text = "\n\n".join(context_parts)

                # Run an LLMChain using the QA prompt
                qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
                answer_text = qa_chain.run(context=context_text, question=query)
            
            except Exception as e:
                print("Error generating answer:", e)
                answer_text = "I don't have information about that."

        # Step 2: Fallback to Search if Local failed or answer is "I don't know"
        # We check specific phrases that indicate failure
        failure_phrases = [
            "I don't have information about that",
            "I don't have info",
            "not mentioned in the context",
            "context does not contain",
            "sorry"
        ]
        
        is_failure = not docs or any(phrase.lower() in answer_text.lower() for phrase in failure_phrases)
        
        # Only search if it WAS relevant but local DB failed
        if is_failure:
            print(f"Local info insufficient. Searching web for: {rewritten_query} (College: {COLLEGE_NAME})")
            try:
                # Add college context to search
                search_query = f"{rewritten_query} {COLLEGE_NAME}"
                search = DuckDuckGoSearchRun()
                web_results = search.run(search_query)
                
                context_text = f"Source: Web Search (DuckDuckGo)\n\n{web_results}"
                
                # Run QA chain again with web results
                # We need a strict prompt here too to avoid hallucination if search is garbage
                qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
                answer_text = qa_chain.run(context=context_text, question=query)
                
                # If even web search fails (returns "I don't have info"), then we truly fail.
                # But usually DuckDuckGo returns something.
                
                # Create a mock document for the source
                web_doc = Document(page_content=web_results, metadata={'source': 'Web Search (DuckDuckGo)'})
                
                # Update session stats
                self.sessions[session_id]['message_count'] += 1
                
                return {
                    'answer': answer_text,
                    'source_documents': [web_doc],
                    'rewritten_query': rewritten_query
                }
            except Exception as e:
                print(f"Web search failed: {e}")
                answer_text = f"I couldn't find information about that in my database or on the web. Please contact the admissions office at admissions@{COLLEGE_NAME.lower().replace(' ', '')}.edu"
                return {
                    'answer': answer_text,
                    'source_documents': [],
                    'rewritten_query': rewritten_query
                }

        # If we reached here, local answer was successful
        self.sessions[session_id]['message_count'] += 1

        return {
            'answer': answer_text,
            'source_documents': top_docs,
            'rewritten_query': rewritten_query
        }

    def _format_history(self, memory: ConversationBufferMemory) -> str:
        """Format conversation history as string"""
        try:
            messages = memory.chat_memory.messages
            history = []
            for msg in messages[-6:]:  # Last 3 exchanges
                role = "User" if msg.type == "human" else "Assistant"
                history.append(f"{role}: {msg.content}")
            return "\n".join(history)
        except:
            return ""

    def clear_memory(self, session_id: str):
        """Clear conversation memory for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_active_sessions(self) -> int:
        """Get count of active conversation sessions"""
        return len(self.sessions)
