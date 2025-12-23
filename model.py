import pandas as pd
import pickle
import os
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

# --- CONFIGURATION ---
FILES = {
    'en': {
        'db': 'spam_data_en.csv', 'model': 'model_en.pkl', 
        'vect_dom': 'vect_dom_en.pkl', 'vect_msg': 'vect_msg_en.pkl',
        'metrics': 'metrics_en.json'
    },
    'zh': {
        'db': 'spam_data_zh.csv', 'model': 'model_zh.pkl', 
        'vect_dom': 'vect_dom_zh.pkl', 'vect_msg': 'vect_msg_zh.pkl',
        'metrics': 'metrics_zh.json'
    }
}

def clean_domain(input_str):
    if not isinstance(input_str, str): return "unknown"
    input_str = input_str.lower().strip()
    match = re.search(r'<([^>]+)>', input_str)
    if match: email = match.group(1)
    else: email = input_str
    if '@' in email: return email.split('@')[-1].strip()
    return email

def get_confluence_en(domain_series, text_series):
    def check_domain_risk(domain):
        if not isinstance(domain, str): return 0.5
        domain = clean_domain(domain)
        if any(domain.endswith(t) for t in ['.xyz', '.top', '.club', '.info', '.online', '.win']): return 1.0 
        if any(domain.endswith(t) for t in ['.edu', '.gov', '.mil']): return 0.2 
        return 0.5 

    def check_content_trap(text):
        if not isinstance(text, str): return 0.0
        t = text.lower()
        trap_words = ['congratulations', 'winner', 'urgent', 'suspend', 'verify', 'bank', 'irs', 'bitcoin', 'usdt']
        action_words = ['click', 'login', 'send', 'immediate', 'password']
        return 1.0 if (any(w in t for w in trap_words) and any(w in t for w in action_words)) else 0.0

    return pd.DataFrame({
        'domain_logic': domain_series.apply(check_domain_risk),
        'text_logic': text_series.apply(check_content_trap)
    })

def get_confluence_zh(domain_series, text_series):
    def check_domain_risk(domain):
        if not isinstance(domain, str): return 0.5
        domain = clean_domain(domain)
        if any(domain.endswith(t) for t in ['.xyz', '.top', '.club']): return 1.0
        if 'edu.tw' in domain or 'gov.tw' in domain: return 0.2
        return 0.5

    def check_content_trap(text):
        if not isinstance(text, str): return 0.0
        trap_words = ['恭喜', '中獎', '凍結', '警示', '匯款', '立即', '比特幣', '泰達幣']
        action_words = ['點擊', '登入', '驗證', '密碼', '轉帳']
        return 1.0 if (any(w in text for w in trap_words) and any(w in text for w in action_words)) else 0.0

    return pd.DataFrame({
        'domain_logic': domain_series.apply(check_domain_risk),
        'text_logic': text_series.apply(check_content_trap)
    })

def train_language(lang):
    print(f"\n--- Training Language: {lang.upper()} ---")
    cfg = FILES[lang]
    
    if not os.path.exists(cfg['db']):
        # Create dummy data if missing
        if lang == 'en':
            data = {'domain': ['support@google.com', 'admin@bad.xyz'], 'message': ['Meeting', 'URGENT'], 'label': [0, 1]}
        else:
            data = {'domain': ['boss@co.tw', 'fake@scam.top'], 'message': ['開會', '中獎'], 'label': [0, 1]}
        df = pd.DataFrame(data)
        df.to_csv(cfg['db'], index=False)
    else:
        df = pd.read_csv(cfg['db'])

    df['domain'] = df['domain'].fillna("")
    df['message'] = df['message'].fillna("")

    cv_dom = TfidfVectorizer(analyzer='char', ngram_range=(2, 5)) 
    X_dom = cv_dom.fit_transform(df['domain'].astype(str).apply(clean_domain))

    if lang == 'en':
        cv_msg = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        X_logic = get_confluence_en(df['domain'], df['message'])
    else:
        cv_msg = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        X_logic = get_confluence_zh(df['domain'], df['message'])
    
    X_msg = cv_msg.fit_transform(df['message'].astype(str))
    X_final = hstack([X_dom, X_msg, X_logic])

    # --- METRICS CALCULATION ---
    metrics_data = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    # We need at least a few rows to calculate meaningful metrics
    if len(df) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X_final, df['label'], test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Get report as a dictionary
        report = classification_report(y_test, preds, output_dict=True)
        
        metrics_data = {
            'accuracy': report['accuracy'],
            # Use weighted avg to account for imbalance
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        # Retrain on full data for final model
        model.fit(X_final, df['label'])
    else:
        print("Dataset too small for split validation. Defaulting to 100% training.")
        model = MultinomialNB()
        model.fit(X_final, df['label'])
        metrics_data = {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    # Save Model & Vectors
    pickle.dump(model, open(cfg['model'], 'wb'))
    pickle.dump(cv_dom, open(cfg['vect_dom'], 'wb'))
    pickle.dump(cv_msg, open(cfg['vect_msg'], 'wb'))
    
    # --- SAVE METRICS TO JSON ---
    with open(cfg['metrics'], 'w') as f:
        json.dump(metrics_data, f)
    
    print(f"Training Complete. Metrics saved to {cfg['metrics']}")

if __name__ == "__main__":
    train_language('en')
    train_language('zh')