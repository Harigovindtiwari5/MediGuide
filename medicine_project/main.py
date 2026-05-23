from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback
import sys

app = Flask(__name__)

BASE = Path(__file__).resolve().parent

def load_csv(name):
    """Load CSV from multiple possible locations"""
    p1 = BASE / name
    p2 = BASE / "datasets" / name
    if p1.exists():
        return pd.read_csv(p1)
    if p2.exists():
        return pd.read_csv(p2)
    raise FileNotFoundError(f"CSV missing: {p1} or {p2}")

def load_model(name):
    """Load model from multiple possible locations"""
    p1 = BASE / name
    p2 = BASE / "models" / name
    if p1.exists():
        return pickle.load(open(str(p1), "rb"))
    if p2.exists():
        return pickle.load(open(str(p2), "rb"))
    raise FileNotFoundError(f"Model missing: {p1} or {p2}")

# ----- Load CSVs -----
try:
    sym_des = load_csv("symtoms_df.csv")
    precautions = load_csv("precautions_df.csv")
    workout = load_csv("workout_df.csv")
    description = load_csv("description.csv")
    medications_df = load_csv("medications.csv")
    diets_df = load_csv("diets.csv")
    training_data = load_csv("Training.csv")
    
    # Load symptom severity weights
    try:
        severity_df = load_csv("Symptom-severity.csv")
        severity_df['Symptom_normalized'] = severity_df['Symptom'].str.lower().str.strip().str.replace(' ', '_')
        symptom_weights = dict(zip(severity_df['Symptom_normalized'], severity_df['weight']))
        print(f"✓ Loaded {len(symptom_weights)} symptom severity weights")
    except:
        symptom_weights = {}
        print("⚠️  symptom-severity.csv not found, using default weights")
        
except FileNotFoundError as e:
    print("FILE LOAD ERROR:", e)
    traceback.print_exc()
    raise

# ----- Load Model -----
try:
    svc = load_model("svc.pkl")
    print(f"✓ Model loaded successfully")
except FileNotFoundError as e:
    print("MODEL LOAD ERROR:", e)
    traceback.print_exc()
    raise

# ----- Determine Feature Names -----
try:
    feature_names = list(training_data.columns[:-1])
    print(f"✓ Loaded {len(feature_names)} feature names")
except Exception as e:
    print("ERROR determining feature names:", e)
    traceback.print_exc()
    raise

feature_index = {name.lower().replace(" ", "_"): i for i, name in enumerate(feature_names)}

# ----- Build Disease Mapping -----
try:
    if hasattr(svc, 'classes_'):
        diseases_list = {i: disease for i, disease in enumerate(svc.classes_)}
        print(f"✓ Loaded {len(diseases_list)} diseases from model")
    else:
        disease_column = training_data.columns[-1]
        unique_diseases = training_data[disease_column].unique()
        diseases_list = {i: disease for i, disease in enumerate(unique_diseases)}
        print(f"✓ Loaded {len(diseases_list)} diseases from training data")
except Exception as e:
    print("ERROR creating disease mapping:", e)
    traceback.print_exc()
    diseases_list = {}

# ----- Enhanced Symptom Synonyms -----
symptom_synonyms = {
    "fever": "high_fever",
    "high fever": "high_fever",
    "temperature": "high_fever",
    
    "headache": "headache",
    "head ache": "headache",
    "head pain": "headache",
    
    "stomach pain": "stomach_pain",
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    
    "body pain": "muscle_pain",
    "body ache": "muscle_pain",
    "aching": "muscle_pain",
    
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    
    "mild fever": "mild_fever",
    "low fever": "mild_fever",
    
    "rash": "skin_rash",
    
    "breathlessness": "breathlessness",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    
    "runny nose": "runny_nose",
    "nose running": "runny_nose",
    
    "congestion": "congestion",
    "blocked nose": "congestion",
    
    "cold hands and feet": "cold_hands_and_feets",
}

def normalize_symptom(symptom):
    """Normalize a symptom string to match feature names"""
    if not symptom:
        return None
    
    symptom = symptom.strip().lower()
    symptom = symptom.replace("severe ", "").replace("mild ", "").replace("chronic ", "")
    
    # Check synonyms first
    if symptom in symptom_synonyms:
        return symptom_synonyms[symptom]
    
    # Try with underscores
    symptom_underscore = symptom.replace(" ", "_")
    if symptom_underscore in feature_index:
        return symptom_underscore
    
    if symptom in feature_index:
        return symptom
    
    # Partial matching
    matches = []
    symptom_words = symptom.split()
    
    for feature in feature_names:
        feature_lower = feature.lower()
        if all(word in feature_lower for word in symptom_words):
            matches.append(feature)
    
    if len(matches) == 1:
        return matches[0].lower().replace(" ", "_")
    elif len(matches) > 1:
        for m in matches:
            parts = m.split("_")
            if symptom in parts or symptom_underscore in parts:
                return m.lower().replace(" ", "_")
        return sorted(matches, key=len)[0].lower().replace(" ", "_")
    
    # Fuzzy matching
    for word in symptom_words:
        if len(word) > 3:
            for feature in feature_names:
                if word in feature.lower():
                    return feature.lower().replace(" ", "_")
    
    return None

def get_predicted_value(patient_symptoms):
    """Predict disease from patient symptoms using weighted features"""
    
    # Create weighted input vector
    input_vector = np.zeros(len(feature_names), dtype=float)
    
    used_features = []
    unmatched = []
    
    # Process each symptom
    for symptom in patient_symptoms:
        if not symptom:
            continue
        
        normalized = normalize_symptom(symptom)
        
        if normalized and normalized in feature_index:
            idx = feature_index[normalized]
            
            # Apply symptom severity weight
            weight = symptom_weights.get(normalized, 3)  # Default weight is 3
            input_vector[idx] = 1 * weight
            
            used_features.append(f"{normalized} (weight={weight})")
        else:
            unmatched.append(symptom.strip())
    
    # Check if any symptoms were matched
    if len(used_features) == 0:
        return ("No symptoms matched. Please use medical symptom terms.", used_features, unmatched)
    
    # Create DataFrame for prediction (model expects this structure)
    # Note: We pass weighted values but column names for sklearn compatibility
    input_df = input_vector.reshape(1, -1)
    
    try:
        # Get prediction
        prediction = svc.predict(input_df)[0]
        
        # Map prediction to disease name
        predicted_disease = None
        
        if isinstance(prediction, str):
            predicted_disease = prediction
        elif isinstance(prediction, (int, np.integer)):
            if prediction in diseases_list:
                predicted_disease = diseases_list[prediction]
            else:
                predicted_disease = str(prediction)
        else:
            if hasattr(svc, 'classes_'):
                try:
                    idx = list(svc.classes_).index(prediction)
                    predicted_disease = svc.classes_[idx]
                except (ValueError, IndexError):
                    predicted_disease = str(prediction)
            else:
                predicted_disease = str(prediction)
        
        # Clean up disease name
        predicted_disease = predicted_disease.strip().replace("_", " ").title()
        
        return (predicted_disease, used_features, unmatched)
        
    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        traceback.print_exc()
        return ("Prediction failed (model error)", used_features, unmatched)

# ----- Flask Routes -----
@app.route("/")
def index():
    return render_template("index.html",
                         predicted_disease=None, 
                         dis_des=None,
                         my_precautions=[], 
                         medications=[],
                         my_diet=[], 
                         workout=[], 
                         message=None)

@app.route("/predict", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        symptoms = request.form.get("symptoms")
        
        if not symptoms or not symptoms.strip():
            message = "Please enter symptoms (comma separated). Example: fever, cough, headache"
            return render_template("index.html", message=message)
        
        # Parse symptoms
        user_symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
        
        # Get prediction
        predicted, used_features, unmatched = get_predicted_value(user_symptoms)
        
        # Handle prediction errors
        if predicted.startswith("No symptoms matched"):
            message = f"No symptoms matched. Try symptoms like: {', '.join(feature_names[:10])}"
            return render_template("index.html", message=message)
        
        if predicted.startswith("Prediction failed"):
            message = "Prediction failed. Please check your input and try again."
            return render_template("index.html", message=message)
        
        predicted_disease = predicted
        
        # Get disease description
        try:
            dis_df = description[description["Disease"].str.strip().str.lower() == predicted_disease.lower()]
            
            if len(dis_df) == 0:
                disease_with_underscore = predicted_disease.replace(" ", "_")
                dis_df = description[description["Disease"].str.strip().str.lower() == disease_with_underscore.lower()]
            
            dis_des = " ".join(dis_df["Description"].values) if len(dis_df) > 0 else "Description not available."
        except Exception as e:
            print(f"Description error: {e}")
            dis_des = "Description not available."
        
        # Get precautions
        my_precautions = []
        try:
            pre = precautions[precautions["Disease"].str.strip().str.lower() == predicted_disease.lower()]
            
            if len(pre) == 0:
                disease_with_underscore = predicted_disease.replace(" ", "_")
                pre = precautions[precautions["Disease"].str.strip().str.lower() == disease_with_underscore.lower()]
            
            if len(pre) > 0:
                row = pre.iloc[0]
                for i in range(1, 5):
                    precaution = row.get(f"Precaution_{i}", "")
                    if str(precaution).strip() and str(precaution).lower() != "nan":
                        my_precautions.append(str(precaution).strip())
        except Exception as e:
            print(f"Precautions error: {e}")
        
        # Get medications
        medications_list = []
        try:
            med = medications_df[medications_df["Disease"].str.strip().str.lower() == predicted_disease.lower()]
            
            if len(med) == 0:
                disease_with_underscore = predicted_disease.replace(" ", "_")
                med = medications_df[medications_df["Disease"].str.strip().str.lower() == disease_with_underscore.lower()]
            
            if len(med) > 0:
                medications_list = [str(m).strip() for m in med["Medication"].values if str(m).strip() and str(m).lower() != "nan"]
        except Exception as e:
            print(f"Medications error: {e}")
        
        # Get diet
        my_diet_list = []
        try:
            diet = diets_df[diets_df["Disease"].str.strip().str.lower() == predicted_disease.lower()]
            
            if len(diet) == 0:
                disease_with_underscore = predicted_disease.replace(" ", "_")
                diet = diets_df[diets_df["Disease"].str.strip().str.lower() == disease_with_underscore.lower()]
            
            if len(diet) > 0:
                import ast
                raw_diets = []
                for d in diet["Diet"].values:
                    d_str = str(d).strip()
                    if d_str and d_str.lower() != "nan":
                        try:
                            parsed = ast.literal_eval(d_str)
                            if isinstance(parsed, list):
                                raw_diets.extend([str(item).strip() for item in parsed])
                            else:
                                raw_diets.append(d_str)
                        except Exception:
                            raw_diets.append(d_str)
                my_diet_list = raw_diets
        except Exception as e:
            print(f"Diet error: {e}")
        
        # Get workout
        workout_list = []
        try:
            wrk_col = "disease" if "disease" in workout.columns else "Disease"
            wrk = workout[workout[wrk_col].str.strip().str.lower() == predicted_disease.lower()]
            
            if len(wrk) == 0:
                disease_with_underscore = predicted_disease.replace(" ", "_")
                wrk = workout[workout[wrk_col].str.strip().str.lower() == disease_with_underscore.lower()]
            
            if len(wrk) > 0:
                workout_list = [str(w).strip() for w in wrk["workout"].values if str(w).strip() and str(w).lower() != "nan"]
        except Exception as e:
            print(f"Workout error: {e}")
        
        # Build message
        message = None
        if len(unmatched) > 0:
            message = f"Note: Some symptoms not recognized: {', '.join(unmatched)}. Used: {', '.join([f.split('(')[0].strip() for f in used_features])}"
        elif len(used_features) > 0:
            message = f"Analyzed {len(used_features)} weighted symptoms"
        
        return render_template("index.html",
                             predicted_disease=predicted_disease,
                             dis_des=dis_des,
                             my_precautions=my_precautions,
                             medications=medications_list,
                             my_diet=my_diet_list,
                             workout=workout_list,
                             message=message)
    
    return render_template("index.html")

@app.route("/debug")
def debug():
    """Debug endpoint"""
    output = []
    output.append(f"Model: {type(svc).__name__}")
    output.append(f"Features: {len(feature_names)}")
    output.append(f"Diseases: {len(diseases_list)}")
    output.append(f"Symptom weights loaded: {len(symptom_weights)}")
    output.append("\nFirst 20 features:")
    output.extend(feature_names[:20])
    output.append("\nSample symptom weights:")
    for symptom, weight in list(symptom_weights.items())[:15]:
        output.append(f"  {symptom}: {weight}")
    return "<pre>" + "\n".join(output) + "</pre>"

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Disease Prediction System Starting...")
    print("="*60)
    print(f"✓ Model loaded with {len(feature_names)} features")
    print(f"✓ {len(diseases_list)} diseases in database")
    print(f"✓ Using symptom severity weights: {len(symptom_weights) > 0}")
    print("="*60 + "\n")
    app.run(debug=True)