# app.py

from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import numpy as np
import pickle

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import re

os.environ['OPENAI_API_KEY'] = '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'

app = Flask(__name__)

llm_resto = OpenAI(temperature=0.6)
prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 6 restaurants names, 6 breakfast names, 5 dinner names, and 6 workout names, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}."
)

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == "POST":
        age = request.form['age']
        gender = request.form['gender']
        weight = request.form['weight']
        height = request.form['height']
        veg_or_noveg = request.form['veg_or_nonveg']
        disease = request.form['disease']
        region = request.form['region']
        allergics = request.form['allergics']
        foodtype = request.form['foodtype']

        chain_resto = LLMChain(llm=llm_resto, prompt=prompt_template_resto)
        input_data = {'age': age,
                              'gender': gender,
                              'weight': weight,
                              'height': height,
                              'veg_or_nonveg': veg_or_noveg,
                              'disease': disease,
                              'region': region,
                              'allergics': allergics,
                              'foodtype': foodtype}
        results = chain_resto.run(input_data)

        # Extracting the different recommendations using regular expressions
        restaurant_names = re.findall(r'Restaurants:(.*?)Breakfast:', results, re.DOTALL)
        breakfast_names = re.findall(r'Breakfast:(.*?)Dinner:', results, re.DOTALL)
        dinner_names = re.findall(r'Dinner:(.*?)Workouts:', results, re.DOTALL)
        workout_names = re.findall(r'Workouts:(.*?)$', results, re.DOTALL)

        # Cleaning up the extracted lists
        restaurant_names = [name.strip() for name in restaurant_names[0].strip().split('\n') if name.strip()] if restaurant_names else []
        breakfast_names = [name.strip() for name in breakfast_names[0].strip().split('\n') if name.strip()] if breakfast_names else []
        dinner_names = [name.strip() for name in dinner_names[0].strip().split('\n') if name.strip()] if dinner_names else []
        workout_names = [name.strip() for name in workout_names[0].strip().split('\n') if name.strip()] if workout_names else []

        return render_template('dietresult.html', restaurant_names=restaurant_names,breakfast_names=breakfast_names,dinner_names=dinner_names,workout_names=workout_names)
    return render_template('index.html')

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    
    if username == "nihal" and password == "team10":
        return redirect(url_for('profile'))
    else:
        return render_template("login.html", error="Invalid credentials")
    
@app.route("/bookint")
def booking():
    return render_template("booking.html")

@app.route("/gameroute")
def gameroute():
    return render_template("gameroute.html")

@app.route("/demomail")
def demomail():
    return render_template("demomail.html")

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/game")
def game():
    return render_template("game.html")

@app.route("/game2")
def game2():
    return render_template("game2.html")
    
@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/home", methods=['POST'])
def home():
    return render_template("home.html")

@app.route("/chatbot")   
def chatbot():
    return render_template("chatbot.html")

@app.route("/result", methods=['POST'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join('models\scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('models\dt.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')
    

@app.route("/dietindex")   
def dietindex():
    return render_template("dietindex.html")


if __name__ == "__main__":
    app.run(debug=True, port=7384)

