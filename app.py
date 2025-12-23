from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import pandas as pd
import time
import json
import os
from scipy.sparse import hstack
import model

app = Flask(__name__)
app.secret_key = 'verimail_secret_key_123'

MODELS = {}

def load_resources():
    for lang in ['en', 'zh']:
        try:
            MODELS[f'{lang}_model'] = pickle.load(open(f'model_{lang}.pkl', 'rb'))
            MODELS[f'{lang}_vect_dom'] = pickle.load(open(f'vect_dom_{lang}.pkl', 'rb'))
            MODELS[f'{lang}_vect_msg'] = pickle.load(open(f'vect_msg_{lang}.pkl', 'rb'))
        except:
            model.train_language(lang)
            MODELS[f'{lang}_model'] = pickle.load(open(f'model_{lang}.pkl', 'rb'))
            MODELS[f'{lang}_vect_dom'] = pickle.load(open(f'vect_dom_{lang}.pkl', 'rb'))
            MODELS[f'{lang}_vect_msg'] = pickle.load(open(f'vect_msg_{lang}.pkl', 'rb'))

load_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        time.sleep(1.5)
        
        domain_input = request.form['domain']
        message_input = request.form['message']
        lang = request.form.get('language', 'en')
        
        # 1. Clean Domain Input
        clean_dom = model.clean_domain(domain_input)
        
        # 2. Select Language Engines
        clf = MODELS[f'{lang}_model']
        v_dom = MODELS[f'{lang}_vect_dom']
        v_msg = MODELS[f'{lang}_vect_msg']
        
        # 3. Calculate Logic
        if lang == 'zh':
            logic_df = model.get_confluence_zh(pd.Series([clean_dom]), pd.Series([message_input]))
        else:
            logic_df = model.get_confluence_en(pd.Series([clean_dom]), pd.Series([message_input]))

        # 4. Vectorize Inputs
        vect_d = v_dom.transform([clean_dom])
        vect_m = v_msg.transform([message_input])
        
        # 5. Combine
        final_input = hstack([vect_d, vect_m, logic_df])
        
        # --- PREDICTION & PROBABILITY CALCULATION ---
        # Get binary prediction
        prediction_array = clf.predict(final_input)
        prediction = prediction_array[0]
        
        # Get Probability (The percentage)
        probs = clf.predict_proba(final_input)[0]
 
        confidence = probs[prediction] 
        final_percentage = round(confidence * 100, 1)

        metrics_file = f'metrics_{lang}.json'
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            except:
                print(f"Could not read {metrics_file}")

        return render_template('result.html', 
                             prediction=prediction, 
                             prob_score=final_percentage,
                             domain=domain_input, 
                             message=message_input, 
                             lang=lang,
                             metrics=metrics)

@app.route('/teach', methods=['POST'])
def teach():
    domain = request.form['domain']
    message = request.form['message']
    label = int(request.form['label'])
    lang = request.form.get('language', 'en')
    
    file_path = model.FILES[lang]['db']
    
    df = pd.read_csv(file_path)
    new_row = pd.DataFrame({'domain': [domain], 'message': [message], 'label': [label]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)
    
    model.train_language(lang)
    load_resources()
    
    msg = "系統已學習！" if lang == 'zh' else "System Updated! Domain memory refreshed."
    flash(msg) 
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)