# -------------------------------------------------------------------------
# AUTHOR:  Anupama Singh
# FILENAME: similarity.py
# SPECIFICATION: ''' This program calculates the cosine similarity between two vectors using basic Python constructs such as lists, dictionaries, and arrays. 
#                    It computes the dot product and magnitudes of vectors to determine their cosine similarity, 
#                    which is a measure of similarity between two non-zero vectors in an inner product space.'''
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 2.5 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv

def tokenize(text):
    '''This function breaks down the text into individual words (tokens) in lowercase.'''
    return text.lower().split()

# Initializing lists to store document IDs and their corresponding tokenized text
documents = []
doc_ids = []

# Specifying the path to the CSV file that contains the documents
file_path = 'cleaned_documents.csv'

# Try block to handle file reading errors and ensure the file is found in the directory
try:
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # # Skip the header of the CSV file
        for row in reader:
            doc_ids.append(int(row[0]))  # Storing the document ID
            documents.append(tokenize(row[1]))  # Tokenizing the document text
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Ensure it is in the correct directory.")
    exit()

# Step 2: Build the document-term matrix
# Extracting all unique words from all the documents
unique_words = set(word for doc in documents for word in doc)  # All unique words
unique_words = sorted(unique_words)  # Sort for consistency
word_index = {word: i for i, word in enumerate(unique_words)}  # Word to index mapping

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here

doc_term_matrix = []

for doc in documents:
    # Creating a binary vector where each word is represented as 1 (if it exists) or 0 (if it doesn't)
    vector = [0] * len(unique_words)
    for word in doc:
        vector[word_index[word]] = 1  # Marking the presence of the word
    doc_term_matrix.append(vector)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
# Function to compute cosine similarity between two vectors
def sqrt(value):
    '''Manual square root calculation using the Newton-Raphson method.'''
    x = value
    y = (x + 1) / 2
    while abs(x - y) > 1e-10:
        x = y
        y = (x + value / x) / 2
    return x

def cosine_similarity(vec1, vec2):
    '''Computes cosine similarity between two vectors.'''
    # Dot product of the two vectors
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    # Calculating the magnitudes (norms) of the vectors
    norm1 = sqrt(sum(v ** 2 for v in vec1))
    norm2 = sqrt(sum(v ** 2 for v in vec2))
    # Returning the cosine similarity: dot product divided by the product of magnitudes
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
# Initializing variables to keep track of the maximum cosine similarity and the most similar documents
max_similarity = 0
most_similar_docs = (None, None)

# Comparing the documents pairwise to find the highest cosine similarity
for i in range(len(doc_term_matrix)):
    for j in range(i + 1, len(doc_term_matrix)):  # Avoid comparing the same document
        sim = cosine_similarity(doc_term_matrix[i], doc_term_matrix[j])
        # Printing out the current comparison and similarity score
        print(f"Comparing document {i+1} and {j+1} with similarity: {sim}")

        # Updating the most similar documents if a higher similarity score is found
        if sim > max_similarity:
            max_similarity = sim
            most_similar_docs = (doc_ids[i], doc_ids[j])


# Step 5: Displaying the most similar documents along with their similarity score
print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} with cosine similarity = {max_similarity:.4f}")