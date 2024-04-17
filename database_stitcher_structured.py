import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
# import tinysegmenter  # For tokenizing Japanese text
#from openai import OpenAI


# Set your OpenAI API key here
openai.api_key = 'key'

data1 = pd.read_csv('data1.csv')
data3 = pd.read_csv('data3.csv')

# Preprocessing as before
data1['search_field'] = data1[['category', 'description', 'id item', 'item yazaki pin', 'catalog type', 'cavity family', 'color', 'gender', 'hybrid poles', 'locking', 'pin rows layout', 'poles open', 'sealed', 'temperature max', 'temperature min', 'terminal size', 'total poles']].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)
data3['search_field'] = data3[['category', 'description', 'id item', 'item yazaki pin', 'catalog type', 'cavity family', 'color', 'gender', 'hybrid poles', 'locking', 'pin rows layout', 'poles open', 'sealed', 'temperature max', 'temperature min', 'terminal size', 'total poles']].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)

vectorizer1 = TfidfVectorizer(stop_words='english')
tfidf_matrix1 = vectorizer1.fit_transform(data1['search_field'])

vectorizer3 = TfidfVectorizer(stop_words='english')
tfidf_matrix3 = vectorizer3.fit_transform(data3['search_field'])


def search(query, previous_results=None):
    if previous_results is not None:
        data_subset = previous_results
    else:
        data_subset = data1

    query_vec1 = vectorizer1.transform([query.lower()])
    query_vec3 = vectorizer3.transform([query.lower()])
    similarity1 = cosine_similarity(query_vec1,tfidf_matrix1[data1.index if previous_results is None else data_subset.index])
    similarity3 = cosine_similarity(query_vec3,tfidf_matrix3[data3.index if previous_results is None else data_subset.index])
    similarity = similarity1 + similarity3
    scores = similarity.flatten()
    filtered_indices = scores.argsort()[::-1]
    return data_subset.iloc[filtered_indices]

# Main chatbot loop
search_results = None # fill w options
print("Hi! I am Akai, Yazaki's AI Chatbot!")
while True:
    user_input = input("Please describe what you're looking for (or type 'e' to exit): ")
    if user_input.lower() == 'e':
        print("Thank you for using the search bot. Goodbye!")
        break
    
    search_results = search(user_input, search_results if search_results is not None else None)

    # Generate a textual description for the top result
    if not search_results.empty:
        print("Top matching product(s) based on your query:\n")
        print(search_results.head(3)['id item'])
    else:
        print("No products found matching your query.")


    refine_search = input("Would you like to refine your search? (yes/no): ")
    if refine_search.lower() != 'yes':
        break
