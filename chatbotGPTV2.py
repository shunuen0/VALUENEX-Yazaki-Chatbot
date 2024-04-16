import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import numpy as np

# Set your OpenAI API key here
openai.api_key = 'key'

data = pd.read_csv('path')
data['search_field'] = data[['category', 'description', 'id item', 'item yazaki pin', 'catalog type', 'cavity family', 'color', 'gender', 'hybrid poles', 'locking', 'pin rows layout', 'poles open', 'sealed', 'temperature max', 'temperature min', 'terminal size', 'total poles']].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['search_field'])

def search(query, previous_results=None):
    if previous_results is not None:
        data_subset = previous_results
    else:
        data_subset = data

    query_vec = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vec, tfidf_matrix[data.index if previous_results is None else data_subset.index])
    scores = similarity.flatten()
    
    # Apply a threshold to filter out low similarity scores
    threshold = 0.1  # Example threshold, adjust as necessary
    relevant_indices = [index for index, score in enumerate(scores) if score > threshold]

    # Sort indices by scores in descending order
    sorted_relevant_indices = sorted(relevant_indices, key=lambda i: scores[i], reverse=True)

    # Return the filtered and sorted results
    matching_results = data_subset.iloc[sorted_relevant_indices]
    return matching_results, len(matching_results)

def identify_key_differentiators(top_results):
    """
    Identifies key differentiating attributes among the top search results.
    Attributes that significantly vary among top results are selected.
    """
    # Thresholds for identifying key differentiators
    unique_threshold = 0.5  # For categorical data, more than 50% unique
    variance_threshold = 0.1 * np.mean(top_results.select_dtypes(include=[np.number]).max() - top_results.select_dtypes(include=[np.number]).min())  # 10% of the range for numeric data
    
    differentiators = []

    # Check numeric fields for high variance
    numeric_data = top_results.select_dtypes(include=[np.number])
    for column in numeric_data:
        if numeric_data[column].var() > variance_threshold:
            differentiators.append(column)

    # Check categorical fields for high uniqueness
    categorical_data = top_results.select_dtypes(include=[object])
    for column in categorical_data:
        unique_count = categorical_data[column].nunique()
        if unique_count / len(categorical_data[column]) > unique_threshold:
            differentiators.append(column)

    return differentiators


def generate_clarifying_questions(top_results):
    differentiators = identify_key_differentiators(top_results)
    questions = []
    for attr in differentiators:
        unique_values = top_results[attr].unique()
        if len(unique_values) > 1:
            question = f"What is your preference for {attr}? Options: {', '.join(map(str, unique_values))}."
            questions.append(question)
    return questions[:-1]

# Main chatbot loop
#search_results = None  # fill w options
print("Hi! I am Akai, Yazaki's AI Chatbot!")
while True:
    user_input = input("Please describe what you're looking for (or type 'e' to exit): ")
    if user_input.lower() == 'e':
        print("Thank you for using the search bot. Goodbye!")
        break    
    search_results, count = search(user_input)

    # Check if results were found and display them
    if count > 0:
        print(f"There are {count} products that match your description.")
        print("Top matching product(s) based on your query:\n")
        for index, row in search_results.head(3).iterrows():
            print(f"Row {index}:")
            for column in search_results.columns[:-1]:
                print(f"{column}: {row[column]}")
            print()
    else:
        print("No products found matching your query.")
    
    print("Below are some clarifying questions that might refine your search \n")
    top_results = search_results.head(3)  # Adjust number of top results as needed
    clarifying_questions = generate_clarifying_questions(top_results)
    for question in clarifying_questions:
        print(question)
    refine_search = input("\nWould you like to refine your search? (yes/no): ")
    if refine_search.lower() != 'yes':
        break
