from flask import Flask, request, render_template
import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Directory containing PDFs
PDF_DIRECTORY = '.'  # Update this to the actual path where your PDFs are stored

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

def extract_and_preprocess_pdfs(directory):
    pdf_processed_words = {}
    for pdf_file in os.listdir(directory):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(directory, pdf_file)
            words = extract_words_from_pdf(pdf_path)
            processed_words = preprocess_words(words)
            pdf_processed_words[pdf_file] = processed_words  # Store preprocessed words for each PDF
    return pdf_processed_words

def compute_tf(doc_words):
    tf = Counter(doc_words)
    total_words = len(doc_words)
    tf = {word: count / total_words for word, count in tf.items()}  # Normalize by document length
    return tf

def compute_idf(documents):
    N = len(documents)  # Total number of documents
    idf = {}
    all_words = set([word for words in documents.values() for word in words])
    for word in all_words:
        # Count the number of documents containing the word
        doc_count = sum(1 for doc_words in documents.values() if word in doc_words)
        idf[word] = math.log(N / (1 + doc_count))  # Smoothed IDF
    return idf

def compute_tfidf(tf, idf):
    tfidf = {word: tf_val * idf[word] for word, tf_val in tf.items() if word in idf}
    return tfidf

def build_tfidf_matrix(documents, idf):
    tfidf_matrix = {}
    for doc, words in documents.items():
        tf = compute_tf(words)
        tfidf = compute_tfidf(tf, idf)
        tfidf_matrix[doc] = tfidf
    return tfidf_matrix

def compute_cosine_similarity(query_vector, document_vectors):
    query_vector = np.array(query_vector)
    similarities = {}
    for doc, vector in document_vectors.items():
        doc_vector = np.array(vector)
        # Compute cosine similarity
        cos_sim = cosine_similarity([query_vector], [doc_vector])[0][0]
        similarities[doc] = cos_sim
    return similarities

def rank_documents(query, pdf_processed_words):
    # Preprocess query
    query_words = preprocess_words(word_tokenize(query))
    
    # Compute TF for the query
    query_tf = compute_tf(query_words)
    
    # Compute IDF for the corpus (including query terms)
    idf = compute_idf(pdf_processed_words)
    
    # Compute TF-IDF for query
    query_tfidf = compute_tfidf(query_tf, idf)
    
    # Build TF-IDF matrix for documents
    tfidf_matrix = build_tfidf_matrix(pdf_processed_words, idf)
    
    # Ensure all vectors have the same length (add missing terms as 0)
    all_words = set([word for tfidf in tfidf_matrix.values() for word in tfidf])
    query_vector = [query_tfidf.get(word, 0) for word in all_words]
    
    # Convert document TF-IDF to vectors of the same length
    document_vectors = {doc: [tfidf.get(word, 0) for word in all_words] for doc, tfidf in tfidf_matrix.items()}
    
    # Compute Cosine Similarity between query and documents
    similarities = compute_cosine_similarity(query_vector, document_vectors)
    
    # Rank documents by similarity
    ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_docs

# Flask routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')

    if not query:
        return render_template('index.html', error='No query provided')

    # Preprocess PDFs from the system directory and rank based on the query
    pdf_processed_dict = extract_and_preprocess_pdfs(PDF_DIRECTORY)
    ranked_docs = rank_documents(query, pdf_processed_dict)

    return render_template('index.html', query=query, ranked_docs=ranked_docs)

if __name__ == "__main__":
    app.run(debug=True)
