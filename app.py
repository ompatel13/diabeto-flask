from flask import Flask, render_template, request
import joblib
import google.generativeai as genai
import markdown 
import os
import pandas

ml_model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai_model = genai.GenerativeModel('gemini-2.0-flash')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        name = request.form['name']
        age = int(request.form['Age'])
        diet_pref = request.form['diet_preference']
        state = request.form['state']
        country = request.form['country']

        features = [float(request.form.get(f)) for f in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]

        
        bmi = features[5]
        glucose = features[1]

        
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = pd.DataFrame([features], columns=columns)
        scaled_features = scaler.transform(input_df)
        prediction = ml_model.predict(scaled_features)[0]
        prob = ml_model.predict_proba(scaled_features)[0][1]

        
        if prediction == 1:
            result = f"You are at risk of Diabetes. (Confidence: {prob*100:.2f}%)"
            return render_template('result.html', result=result, risk=True, user_data={
                "name": name,
                "age": age,
                "diet_preference": diet_pref,
                "state": state,
                "country": country,
                "bmi_value": bmi,
                "glucose_value": glucose
            })
        else:
            result = f"You are not at risk of Diabetes. (Confidence: {(1 - prob)*100:.2f}%)"
            return render_template('result.html', result=result, risk=False)

    except Exception as e:
        return f"Error: {e}"

@app.route('/diet', methods=['POST'])
def diet():
    try:
        user_data = {
            "name": request.form.get("name"),
            "age": request.form.get("age"),
            "diet_preference": request.form.get("diet_preference"),
            "state": request.form.get("state"),
            "country": request.form.get("country"),
            "bmi_value": request.form.get("bmi"),
            "glucose_value": request.form.get("glucose")
        }

        prompt = f"""
        A {user_data['age']}-year-old {user_data['diet_preference']} woman named {user_data['name']} from {user_data['state']}, {user_data['country']}, has a BMI of {user_data['bmi_value']} and a glucose level of {user_data['glucose_value']}. Write a 5-point personalized diabetes-prevention diet plan for her, explaining why each point is important not to long but not to short and if name any food then in give name of it in local language in (), without adding unnecessary information and reponse like you are talking to {user_data['name']} in friendly tone and simple language.
        """

        response = genai_model.generate_content(prompt)

        diet_plan_html = markdown.markdown(response.text)

        return render_template("diet.html", name=user_data["name"], diet_plan=diet_plan_html)

    except Exception as e:
        return f"Gemini API Error: {e}"



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

