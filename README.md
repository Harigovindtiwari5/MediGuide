
# MediGuide 🏥
### Disease Prediction and Medicine Recommendation System

MediGuide is a machine learning-based web application that predicts diseases based on user-reported symptoms and provides complete health guidance including medicines, precautions, diet plans, and workout recommendations.

---

## ✨ Features

- 🔍 **Disease Prediction** — Predicts disease from symptoms using SVM (Support Vector Machine)
- 💊 **Medicine Recommendations** — Suggests relevant medicines for predicted disease
- ⚠️ **Precautions** — Lists important precautions to follow
- 🥗 **Diet Plan** — Recommends diet based on the disease
- 🏃 **Workout Suggestions** — Provides suitable workout/exercise advice
- ⚡ **Real-time Results** — Instant predictions with weighted symptom analysis
- 📱 **Responsive UI** — Works on both desktop and mobile

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.x |
| Framework | Flask |
| ML Model | Support Vector Machine (SVM) |
| Frontend | HTML5, CSS3, Bootstrap 5 |
| Libraries | Scikit-learn, Pandas, NumPy |
| Dataset | 132 symptoms, 41 diseases, 4920 records |

---

## 📁 Project Structure

```
MediGuide/
├── main.py                  # Flask application
├── train_model.py           # Model training script
├── svc.pkl                  # Trained SVM model
├── templates/
│   ├── index.html           # Main page
│   ├── about.html
│   ├── contact.html
│   ├── developer.html
│   └── blog.html
├── static/
│   └── logo.png
├── Training.csv             # Training dataset
├── description.csv          # Disease descriptions
├── medications.csv          # Medicine data
├── precautions_df.csv       # Precautions data
├── diets.csv                # Diet recommendations
├── workout_df.csv           # Workout suggestions
└── Symptom-severity.csv     # Symptom severity weights
```

---

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/harigovindtiwari5/MediGuide.git
cd MediGuide
```

**2. Install dependencies:**
```bash
pip install flask numpy pandas scikit-learn werkzeug
```

**3. Train the model:**
```bash
python train_model.py
```

**4. Run the app:**
```bash
python main.py
```

**5. Open in browser:**
```
http://127.0.0.1:5000
```

---

## 💡 How to Use

1. Enter symptoms in the search box (comma separated)
   - Example: `fever, headache, nausea`
2. Click **Predict Disease**
3. Get complete results:
   - Predicted Disease
   - Description
   - Medicines
   - Precautions
   - Diet Plan
   - Workout Plan

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 99% |
| Test Accuracy | 98% |
| Diseases Covered | 41 |
| Symptoms Supported | 132 |

---

## 👨‍💻 Developer

**Hari Govind Tiwari**  
B.Tech Computer Science and Engineering  
Rajkiya Engineering College, Kannauj  

---

## 📝 Note

> This system is for educational purposes only. Always consult a qualified medical professional for diagnosis and treatment.
