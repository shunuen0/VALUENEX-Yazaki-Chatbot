from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
import openai
import pandas as pd

app = Flask(__name__)
app.config["SECRET_KEY"] = "test"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

openai.api_key = ''  # Use your actual API key here

def read_excel_data(files):
    data_frames = {}
    for file_name, file_path in files.items():
        df = pd.read_excel(file_path)
        # Ensure the loaded data is a DataFrame
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).T  # Convert Series to DataFrame if necessary
        data_frames[file_name] = df
        print(f"Loaded {file_name} with shape {df.shape}")  # Debugging information to check the load
    return data_frames

def generate_prompt(excel_data, chat_history):
    prompt = "Your answers from the excel file should be numerically accurate. For example, if the query says 'less than 3', the answer is 0, 1, and 2 but doesn't include 3.\n"

    # Include the Excel data from all sheets at the beginning of the session or when needed
    for sheet_name, data in excel_data.items():
        prompt += f"\nSheet {sheet_name}:\n"
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        for index, row in data.iterrows():
            for col_name, col_value in row.items():
                prompt += f"{col_name}: {col_value}\n"
        prompt += "\n"
    #prompt += "\n"  # Ensure there is a separation between the Excel data and chat history

    # Add previous conversation history to maintain context, adding a newline after each message
    for role, message in chat_history:
        prompt += f"{role}: {message}\n\n"
    return prompt


def chat_with_rag(user_input):
    excel_data = session.get('excel_data', None)
    chat_history = session.get('chat_history', [])

    chat_history.append(("User", user_input))  # Add user input to history

    total_tokens = sum(len(f"{role}: {message}") for role, message in chat_history)
    if total_tokens > 16385:  # Example threshold, adjust based on your model's limits
        chat_history = []  # Reset the chat history if token count is too high

    prompt = generate_prompt(excel_data, chat_history)
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        temperature = 0,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500
    )
    response_text = response.choices[0].message.content
    chat_history.append(("Akai", response_text))  # Add bot response to history

    session['chat_history'] = chat_history  # Update session

    return response_text

@app.route("/clear_session", methods=["GET"])
def clear_session():
    session.clear()  # Clears the session data
    return redirect(url_for('home'))  # Redirects to the home page

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        if 'excel_data' not in session:
            # Define the paths for each of your Excel files
            excel_files = {
                "Connector Catalog": "",
                "Fixation Catalog": "",
                "Connector Attribute Ranking": "",
                "Fixation Attribute Ranking": "",
                "Output Rules" : ""
            }
            session['excel_data'] = read_excel_data(excel_files)
        response = chat_with_rag(user_input)
        return render_template("index4.html", chat_history=session['chat_history'])
    return render_template("index4.html", chat_history=[])

if __name__ == "__main__":
    app.run(debug=True)
