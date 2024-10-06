import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Initialize the Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to convert POS tags from NLTK to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun if no other tag is found

def extract_words_from_pdf(pdf_path):
    words = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                words += word_tokenize(text)  # Tokenize the text
    return words

def preprocess_words(words):
    # Remove stop words and apply lemmatization
    processed_words = []
    tagged_words = pos_tag(words)  # Part-of-speech tagging
    
    for word, tag in tagged_words:
        word = word.lower()  # Convert to lowercase
        if word.isalpha() and word not in stop_words:  # Remove non-alphabetic tokens and stopwords
            wordnet_pos = get_wordnet_pos(tag)  # Convert to WordNet POS tags
            lemmatized_word = lemmatizer.lemmatize(word, wordnet_pos)  # Apply lemmatization
            processed_words.append(lemmatized_word)
    return processed_words

def extract_and_preprocess_pdfs(root_dir):
    pdf_processed_words = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                words = extract_words_from_pdf(pdf_path)
                processed_words = preprocess_words(words)
                pdf_processed_words[file] = processed_words  # Store preprocessed words for each PDF
    return pdf_processed_words

def build_inverted_index(pdf_processed_words):
    inverted_index = {}

    for pdf_file, words in pdf_processed_words.items():
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []  # Initialize the list for the word
            if pdf_file not in inverted_index[word]:
                inverted_index[word].append(pdf_file)  # Add PDF file name to the word's list
    
    return inverted_index

if __name__ == "__main__":
    root_dir = '.'  # Root directory of the project
    pdf_processed_dict = extract_and_preprocess_pdfs(root_dir)

    # Build the inverted index from preprocessed words
    inverted_index = build_inverted_index(pdf_processed_dict)

    # Display the inverted index (show only part of it for readability)
    for word, pdf_files in list(inverted_index.items()):  # Show first 10 entries
        print(f"Word: {word}, PDFs: {pdf_files}")
