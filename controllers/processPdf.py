import os
import json
import io
import time
import tempfile
import numpy as np
import csv

import tabula
# from PIL import Image
from PyPDF2 import PdfReader
from pypdf import PdfReader as ImageReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

import google.generativeai as genai

from utils.modelSettings import generation_config, safety_settings
# from utils.modelInstructions import image_retriever_instruction
from utils.geminiUtils import GeminiUtils

# Configure the API key for Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class PDFprocessor:
    """
    A class for processing PDF documents, extracting text, tables and generating summaries.
    """

    def __init__(self, pdf_docs, pdf_names):
        """
        Initialize the PDF processor with provided PDF documents and names.

        Args:
            pdf_docs (list): List of PDF documents.
            pdf_names (list): List of PDF document names.
        """
        self.pdf_docs = pdf_docs
        self.pdf_names = pdf_names

    """
        MAIN FUNCTION 1 : parse_pdf()
        - Parses PDF for text, converts texts to chunks
        - Uses functions listed below:
            - get_pdf_text()
            - get_text_chunks()
    """

    def parse_pdf(self):
        """
        Parses the PDF to get text and then splits it into chunks.

        Returns:
            list: List of text chunks.
        """
        text_with_metadata = self.get_pdf_text()
        chunks = self.get_text_chunks(
            text_with_metadata, type="text", csv_name="")
        return chunks

    """
        MAIN FUNCTION 2 : get_tables()
        - Parses PDF for tables, converts all tables to csv and then to chunks
        - Uses functions listed below:
            - get_csv_string()
            - filter_csv()
            - get_text_chunks()
    """

    def get_tables(self):
        """
        Extracts tables from the PDFs and splits them into chunks.

        Returns:
            list: List of text chunks from tables.
        """
        csv_chunks = []
        for pdf, pdf_name in zip(self.pdf_docs, self.pdf_names):
            tabula.convert_into(pdf, "context_docs/output.csv",
                                output_format="csv", pages="all", lattice=True)

            self.filter_csv("context_docs/output.csv",
                            "context_docs/output_filtered.csv")

            csv_string = self.get_csv_string(
                "context_docs/output_filtered.csv")
            csv_chunks.extend(self.get_text_chunks(
                text_with_metadata=csv_string, type="csv", csv_name=pdf_name))
        return csv_chunks

    """
        MAIN FUNCTION 3 : extract_images()
        - Parses PDF for all images, stores them in proper folders
    """

    
    def get_pdf_text(self):
        """
        Extracts text from each page of the PDF with page numbers.

        Returns:
            list: List of dictionaries containing document name, page number, and extracted text.
        """
        try:
            text_with_metadata = []
            for pdf, pdf_name in zip(self.pdf_docs, self.pdf_names):
                pdf_reader = PdfReader(pdf)
                for page_num, page in enumerate(pdf_reader.pages, start=0):
                    text = page.extract_text()
                    if text:
                        text_with_metadata.append(
                            {"doc_name": pdf_name, "page_num": page_num, "text": text})
            return text_with_metadata
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {e}")

    def get_text_chunks(self, text_with_metadata, type, csv_name):
        """
        Splits the raw text into chunks, preserving page metadata.

        Args:
            text_with_metadata (list): List of text with metadata.
            type (str): Type of document (text or csv).
            csv_name (str): Name of the CSV file.

        Returns:
            list: List of text chunks.
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            if type == "csv":
                chunks = text_splitter.create_documents([text_with_metadata])
                for doc in chunks:
                    doc.metadata["doc_name"] = csv_name
                return chunks
            else:
                chunks = []
                for item in text_with_metadata:
                    pdf_name = item["doc_name"]
                    page_num = item["page_num"]
                    raw_text = item["text"]
                    docs = text_splitter.create_documents([raw_text])
                    for doc in docs:
                        doc.metadata["doc_name"] = pdf_name
                        doc.metadata["page_num"] = page_num
                        chunks.append(doc)
                return chunks
        except Exception as e:
            raise RuntimeError(f"Error splitting text: {e}")

    def filter_csv(self, input_csv_path, output_csv_path):
        """
        Filters CSV rows to remove those with insufficient data.

        Args:
            input_csv_path (str): Path to the input CSV file.
            output_csv_path (str): Path to the output filtered CSV file.
        """
        try:
            with open(input_csv_path, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)

            with open(output_csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for row in rows:
                    if sum(1 for column in row if column.strip()) <= 2:
                        continue
                    if any(column.strip() for column in row[1:]):
                        writer.writerow(row)
        except Exception as e:
            raise RuntimeError(f"Error filtering CSV: {e}")

    def get_csv_string(self, input_csv_path):
        """
        Converts a CSV file to a string representation suitable for further processing.

        Args:
            input_csv_path (str): Path to the input CSV file.

        Returns:
            str: String representation of the CSV file.
        """
        try:
            bigstring = []
            with open(input_csv_path, 'r') as f:
                lines = f.readlines()
                smallstring = ""
                for line in lines:
                    new = line.strip()
                    new = new.replace('"', '')
                    smallstring += new
                    if new.find(",,,,,,,,,,,") != -1:
                        bigstring.append(smallstring)
                        smallstring = ""
                bigstring.append(smallstring)

            bigstring_str = repr(bigstring)
            return bigstring_str
        except Exception as e:
            raise RuntimeError(f"Error getting CSV string: {e}")


    def get_relevant_text(self, pdf_base_name, relevant_page_numbers):
        """
        Extracts text from specific pages of a PDF based on provided page numbers.

        Args:
            pdf_base_name (str): Base name of the PDF file.
            relevant_page_numbers (list): List of page numbers to extract text from.

        Returns:
            list: List of text extracted from the specified pages.
        """
        try:
            relevant_text = []
            print(f"Getting relevant text for {pdf_base_name}")
            print(f"Relevant page numbers: {relevant_page_numbers}")
            for pdf, pdf_name in zip(self.pdf_docs, self.pdf_names):
                print(f"Processing {pdf_name} and {pdf_base_name}")

                if (pdf_name == f"{pdf_base_name}.pdf"):
                    pdf_reader = PdfReader(pdf)
                    for page_num, page in enumerate(pdf_reader.pages, start=0):
                        print(f"Processing page {page_num}")
                        if page_num in relevant_page_numbers:
                            text = page.extract_text()
                            relevant_text.append(text)

            return relevant_text
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {e}")
