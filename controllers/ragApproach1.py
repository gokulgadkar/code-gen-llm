import os
import glob
import json

# reg no /name , default value , from the driver code 

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from utils.tracker import store_object, fetch_object
from utils.modelSettings import generation_config, safety_settings
from utils.modelInstructions import (
    answer_retriever_instruction
    
)
from utils.geminiUtils import GeminiUtils
from controllers import processPdf

# Configure the API key for Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Dictionary holding model and store constants for repetitive use
name_constants = {
    "EMBEDDING_MODEL": "models/embedding-001",
    "LLM_TEXT_MODEL": "gemini-1.5-flash-latest",
    "MULTI_MODAL_MODEL": "gemini-1.5-flash-latest",
    "TEXT_EMBEDDING_STORE": "faiss_db/faiss_index",
    "CSV_EMBEDDING_STORE": "faiss_db/csv_faiss_index",
    "TEXT_EMBEDDING_STORE2": "faiss_db/faiss_index2",
    "CSV_EMBEDDING_STORE2": "faiss_db/csv_faiss_index2"
}


class ragApproach1:
    def __init__(self, pdf_docs, chunks, pdf_names, csv_chunks):
        """
        Initialize the RAG approach with provided documents and chunks.

        Args:
            pdf_docs (list): List of PDF documents.
            chunks (list): List of text chunks from the PDFs.
            pdf_names (list): List of PDF document names.
            csv_chunks (list): List of text chunks from CSV files.
        """
        self.pdf_docs = pdf_docs
        self.chunks = chunks
        self.pdf_names = pdf_names
        self.csv_chunks = csv_chunks
        self.model = None
        self.vision_model = None
        self.vision_context_model = None
        self.embeddings = self.initialize_embeddings()

    """
        MAIN FUNCTION 1 : create_vector_store()
        - Converts all provided chunks (textual, tabular chunks) and converts them to embeddings
        - Create a FAISS vector store locally inside faiss_db directory.
        - Uses googles "models/embedding-001" model for embeddings
    """

    def create_vector_store(self):
        """
        Create and save vector stores for document and CSV embeddings.

        Returns:
            bool: True if the stores were created and saved successfully, None otherwise.
        """
        try:
            # Create vector stores for text and CSV embeddings
            store = FAISS.from_documents(self.chunks, self.embeddings)
            csv_store = FAISS.from_documents(self.csv_chunks, self.embeddings)

            # Save vector stores locally
            store.save_local(name_constants["TEXT_EMBEDDING_STORE"])
            csv_store.save_local(name_constants["CSV_EMBEDDING_STORE"])

            return True
        except Exception as e:
            print(e)
            return None

    """
        MAIN FUNCTION 2 : get_answer_for_text_query()
        - Gets answer along with relevant images from the PDFs
        - Converts the query to embeddings
        - Retrieves relevant contexts from the vector stores with similarity search with query embeddings
        - In Approac 1, we get all images from relevant pages surrounding similar chunks.
        - These images are sent to a model along with question, answer and context to get the most appropriate images.
        - Uses functions like:
            - retrieve_contexts() - retrieves relevant contexts from the vector stores
            - format_message_with_context() - formats query and appends to chat history for history aware analysis of model
            - initialize_gen_ai() - initializes the generative AI model with chat history, uses the "gemini-1.5-flash"
            - get_images() - retrieves relevant images from the PDFs from only the relevant pages from where context was pulled
            - call_image_model() - calls the image model for analysis
            - create_json_with_image_data() - creates JSON object with image path and caption of relevant image
        
    """
    def append_to_vector_store(self):
          """
          Append the new document and CSV embeddings to the existing vector store.
          Returns:
              bool: True if the stores were updated and saved successfully, None otherwise.
          """
          try:
              # Load the existing vector stores for text and CSV embeddings
            store = FAISS.from_documents(self.chunks, self.embeddings)
            csv_store = FAISS.from_documents(self.csv_chunks, self.embeddings)

            # Save vector stores locally
            store.save_local(name_constants["TEXT_EMBEDDING_STORE2"])
            csv_store.save_local(name_constants["CSV_EMBEDDING_STORE2"])
      
            return True
          except Exception as e:
            print(e)
            return None

    def get_answer_for_text_query(self, question):
        """
        Generate an answer to the question using the AI model, including images if available.

        Args:
            question (str): The user's question.
            chat_history (list): List of chat history.

        Returns:
            tuple: Generated answer and image data.
        """
        try:

            answers =[]
            que_ext = " Make a table, extract register values"

            for que in question: 
                # Retrieve contexts for the question
                combined_context, csv_combine_context, context_page_info = self.retrieve_contexts(
                    question, name_constants["TEXT_EMBEDDING_STORE"], name_constants["CSV_EMBEDDING_STORE"])
                chat_session = self.initialize_gen_ai(                
                    [
                        {
                        "role": "user",
                        "parts": [
                            self.format_message_with_context(combined_context, csv_combine_context ),
                            que + que_ext,
                        ],
                        }
                    ])
                if not chat_session:
                    return "Failed to initialize chat session."
                # Generate responses from the chat session
                responses = chat_session.send_message(question, stream=True)
                responses.resolve()
                all_responses = [
                    part.text for response in responses for part in response.parts if hasattr(part, 'text')]
                answer = " ".join(all_responses).replace(
                    "``` html", "", 1).replace("\n```", "", 1)
                answers.append(answer)

            return answers
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An error occurred while processing the request."




    def get_answer_for_text_query2(self, question):
        """
        Generate an answer to the question using the AI model, including images if available.

        Args:
            question (str): The user's question.
            chat_history (list): List of chat history.

        Returns:
            tuple: Generated answer and image data.
        """
        try:

            answers =[]
            que_ext = " Make a table, extract register values"

            for que in question: 
                # Retrieve contexts for the question
                combined_context, csv_combine_context, context_page_info = self.retrieve_contexts(
                    question, name_constants["TEXT_EMBEDDING_STORE2"], name_constants["CSV_EMBEDDING_STORE2"])

                chat_session = self.initialize_gen_ai(                
                    [
                        {
                        "role": "user",
                        "parts": [
                            self.format_message_with_context(combined_context, csv_combine_context ),
                            que + que_ext,
                        ],
                        }
                    ])
                if not chat_session:
                    return "Failed to initialize chat session."
                # Generate responses from the chat session
                responses = chat_session.send_message(question, stream=True)
                responses.resolve()
                all_responses = [
                    part.text for response in responses for part in response.parts if hasattr(part, 'text')]
                answer = " ".join(all_responses).replace(
                    "``` html", "", 1).replace("\n```", "", 1)
                answers.append(answer)

            return answers
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An error occurred while processing the request."
            

    """
        MAIN FUNCTION 3 : get_answer_for_text_query()
        - Gets the context for the uploaded image from the PDF
        - We know which image was uploaded from calculate_similarity in processPdf.pdf
        - Get all text from surrounding page of image
        - Get relevant summary of image.
        - Uses functions like:
            - retrieve_relevant_context() - retrieves relevant text from the PDFs
            - call_image_context_model() - calls the image context model for analysis, uses the "gemini-1.5-flash"
    """


    def initialize_embeddings(self):
        """
        Initialize embeddings using the Google Generative AI Embeddings model.

        Returns:
            GoogleGenerativeAIEmbeddings: Initialized embeddings model.
        """
        return GoogleGenerativeAIEmbeddings(model=name_constants["EMBEDDING_MODEL"])

    def initialize_gen_ai(self, chat_history):
        """
        Initialize the generative AI model for text-based interactions.

        Args:
            chat_history (list): List of chat history.

        Returns:
            GenerativeModel: Initialized generative model for chat interactions.
        """
        try:
            if self.model is None:
                self.model = genai.GenerativeModel(
                    model_name=name_constants["LLM_TEXT_MODEL"],
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                    system_instruction=answer_retriever_instruction    
                )
            return self.model.start_chat(history=chat_history)
        except Exception as e:
            print(e)
            return None

    
    def retrieve_contexts(self, question, textConstant, csvConstant):
        """
        Retrieve contexts related to the given question from the vector stores.

        Args:
            question (str): The user's question.

        Returns:
            tuple: Combined context, combined CSV context, and context page information.
        """
        
        store = FAISS.load_local(
            textConstant, self.embeddings)
        csv_store = FAISS.load_local(
            csvConstant, self.embeddings)

        # Retrieve relevant documents
        context_documents = store.similarity_search(question, k=5)
        context_csv_docs = csv_store.similarity_search(question, k=10)

        # Format the retrieved documents
        context_page_info = [
            {'doc_name': doc.metadata.get('doc_name'), 'page_num': int(
                doc.metadata.get('page_num'))}
            for doc in context_documents
        ]
        context_texts = [
            {
                "doc_name": doc.metadata.get('doc_name'),
                "page_num": int(doc.metadata.get('page_num')),
                "content": doc.page_content
            }
            for doc in context_documents
        ]
        context_csv = [
            {
                "doc_name": doc.metadata.get('doc_name'),
                "content": doc.page_content
            }
            for doc in context_csv_docs
        ]

        # Convert lists to JSON strings
        combined_context = json.dumps(context_texts)
        csv_combine_context = json.dumps(context_csv)

        return combined_context, csv_combine_context, context_page_info

    def retrieve_relevant_context(self, relevant_page_numbers, pdf_base_name):
        """ Retrieve image context for the given page numbers.
        Args:
            relevant_page_numbers (list): List of relevant page numbers.
            pdf_base_name (str): Base name of the PDF file.
        Returns:
            str: Image context.
        """
        pPDF = processPdf.PDFprocessor(
            pdf_docs=self.pdf_docs, pdf_names=self.pdf_names)

        relevant_text = pPDF.get_relevant_text(
            pdf_base_name=pdf_base_name, relevant_page_numbers=relevant_page_numbers)

        return relevant_text

    def format_message_with_context(self, combined_context, csv_combine_context):
        """
        Format the retrieved context as a message for the AI model.

        Args:
            combined_context (str): Combined textual context.
            csv_combine_context (str): Combined CSV context.

        Returns:
            dict: Formatted message.
        """
        return f"CONTEXT: {combined_context} \n\n CONTEXT(csv) {csv_combine_context}"
        
