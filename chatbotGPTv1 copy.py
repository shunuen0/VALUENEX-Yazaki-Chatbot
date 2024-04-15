import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
#from openai import OpenAI



# Set your OpenAI API key here
openai.api_key = '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'

data = pd.read_csv('/Users/shunueno/Desktop/Chatbot /dataset/used/Modified_Book1.csv')

# Preprocessing as before
data['search_field'] = data[['category', 'description', 'id item', 'item yazaki pin', 'catalog type', 'cavity family', 'color', 'gender', 'hybrid poles', 'locking', 'pin rows layout', 'poles open', 'sealed', 'temperature max', 'temperature min', 'terminal size', 'total poles']].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['search_field'])

def search(query, previous_results=None):
    if previous_results is not None:
        data_subset = previous_results
    else:
        data_subset = data

    query_vec = vectorizer.transform([query.lower()])
    #similarity = cosine_similarity(query_vec, tfidf_matrix[data_subset.index])
    similarity = cosine_similarity(query_vec,tfidf_matrix[data.index if previous_results is None else data_subset.index])
    scores = similarity.flatten()
    filtered_indices = scores.argsort()[::-1]
    return data_subset.iloc[filtered_indices]

def ask_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content": "You are an component search AI chatbot for Yazaki Corporation"}, {"role":"system","content": prompt}]
    )
    return response.choices[0].message.content

def format_attributes_for_gpt(data_row):
    """
    Format the attributes of a DataFrame row into a descriptive prompt for GPT.
    """
    attributes = []
    for column in data_row.columns:
        value = data_row.iloc[0][column]
        if pd.notna(value):  # Check if the value is not NaN
            attributes.append(f"{column}: {value}")
    attributes_str = ", ".join(attributes)
    return f"Generate a paragraph description for a product with the following attributes: {attributes_str}."


def gpt_create_text_description(row):
    prompt = format_attributes_for_gpt(row)
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"system","content": "Generate a paragraph description based on its attributes. Make sure every attribute is included in the description."}, {"role":"system","content": prompt}]
    )
    return response.choices[0].message.content

# Main chatbot loop
search_results = None # fill w options
print("Hi! I am Akai, Yazaki's AI Chatbot!")
while True:
    user_input = input("Please describe what you're looking for (or type 'e' to exit): ")
    if user_input.lower() == 'e':
        print("Thank you for using the search bot. Goodbye!")
        break
    
    # Use GPT to interpret the query or ask for more details
    gpt_prompt = f"User is looking for a product with the following description: {user_input}. What would be a good clarifying question or search query? Begin your response with 'To better assist you'."
    gpt_response = ask_gpt(gpt_prompt)
    
    # Assuming GPT response is a refined search query for now
    #print(gpt_response)
    print("")
    # Ensure 'search_results' is correctly updated with each call to 'search'
    search_results = search(user_input, search_results if search_results is not None else None)

    # Generate a textual description for the top result
    if not search_results.empty:
        text_description = gpt_create_text_description(search_results.head(1))
        print("Top matching product(s) based on your query:\n")
        #print(search_results.head(1).to_string(index=False))  # Using .to_string(index=False) for cleaner output
        for column in search_results.columns:
            print(f"{column}: {search_results.head(1)[column].values[0]}")
        print("\nGenerated Description:\n", text_description)
    else:
        print("No products found matching your query.")

    print(gpt_response)
    refine_search = input("Would you like to refine your search? (yes/no): ")
    if refine_search.lower() != 'yes':
        break
