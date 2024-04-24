

from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
import openai
import pandas as pd

app = Flask(__name__)
app.config["SECRET_KEY"] = "test"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

openai.api_key = 'key'
prompt = "Your answers from the excel file should be numerically accurate. For example if the query says less than 3, the answer is 0, 1 and 2 but doesn't give me 3.  Additionally, 'waterproof' is a synonym for 'sealed'."
def read_excel_data(file_path):
    return pd.read_excel(file_path)

def generate_prompt(excel_data):
    prompt = "User: {user_input}\n"
    for index, row in excel_data.iterrows():
        for col_name, col_value in row.items():
            prompt += f"{col_name}: {col_value}\n"
    return prompt

def chat_with_rag(user_input, excel_data):
    prompt = generate_prompt(excel_data)
    prompt = prompt.format(user_input=user_input)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

@app.route("/clear_session", methods=["GET"])
def clear_session():
    session.clear()  # Clears the session data
    return redirect(url_for('home'))  # Redirects to the home page

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        if 'chat_history' not in session:
            session['chat_history'] = []
        excel_data = read_excel_data('file')
        response = chat_with_rag(user_input, excel_data)
        session['chat_history'].append(("You", user_input))
        session['chat_history'].append(("Akai", response))
        return render_template("index.html", chat_history=session['chat_history'])
    return render_template("index.html", chat_history=[])

if __name__ == "__main__":
    app.run(debug=True)

    
