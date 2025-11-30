import os
import re
import PyPDF2
import docx

class ResumeParser:
    def __init__(self):
        """Initialize resume parser"""
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_path):
        """
        Extract text from DOCX file
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error reading DOCX {docx_path}: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_path):
        """
        Extract text from TXT file
        
        Args:
            txt_path: Path to TXT file
            
        Returns:
            Extracted text
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
        except Exception as e:
            print(f"Error reading TXT {txt_path}: {str(e)}")
            return ""
    
    def parse_resume(self, file_path):
        """
        Parse resume from file
        
        Args:
            file_path: Path to resume file
            
        Returns:
            Extracted text
        """
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Check if format is supported
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {self.supported_formats}")
        
        # Extract text based on format
        if ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            text = ""
        
        return text
    
    def parse_multiple_resumes(self, folder_path):
        """
        Parse multiple resumes from a folder
        
        Args:
            folder_path: Path to folder containing resumes
            
        Returns:
            Dictionary of {filename: text}
        """
        resumes = {}
        
        # Get all files in folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
            
            # Get extension
            _, ext = os.path.splitext(filename)
            
            # Parse if supported format
            if ext.lower() in self.supported_formats:
                try:
                    text = self.parse_resume(file_path)
                    resumes[filename] = text
                    print(f"Parsed: {filename}")
                except Exception as e:
                    print(f"Error parsing {filename}: {str(e)}")
        
        print(f"\nTotal resumes parsed: {len(resumes)}")
        return resumes
    
    def extract_contact_info(self, text):
        """
        Extract contact information from resume text
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary of contact info
        """
        contact_info = {}
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else None
        
        # Extract phone
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        phones = re.findall(phone_pattern, text)
        contact_info['phone'] = phones[0] if phones else None
        
        # Extract LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text.lower())
        contact_info['linkedin'] = linkedin[0] if linkedin else None
        
        return contact_info
    
    def extract_name(self, text):
        """
        Extract candidate name from resume (simple heuristic)
        
        Args:
            text: Resume text
            
        Returns:
            Candidate name (first line typically)
        """
        lines = text.strip().split('\n')
        # Assume name is in first few lines
        for line in lines[:5]:
            line = line.strip()
            # Simple check: name is typically 2-4 words, all capitalized or title case
            words = line.split()
            if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word):
                return line
        return None