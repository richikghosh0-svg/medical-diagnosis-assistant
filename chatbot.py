"""Challenge 3: offline medical chatbot (TF-IDF retrieval over a small knowledge base)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = [
    {"topic": "Influenza (flu)", "text": "Flu causes sudden fever, cough, sore throat, body aches and fatigue. Rest, fluids and fever reducers help. See a doctor if breathing is difficult, symptoms last over a week, or you are elderly, pregnant or have a chronic illness."},
    {"topic": "Common cold", "text": "A cold causes a runny nose, sneezing, mild cough and sore throat, usually without high fever. It usually resolves in 7-10 days with rest and fluids."},
    {"topic": "COVID-19", "text": "COVID-19 can cause fever, cough, loss of taste or smell, fatigue and breathlessness. Testing is recommended. Isolate while symptomatic and seek care if breathing worsens."},
    {"topic": "Pneumonia", "text": "Pneumonia is a lung infection causing cough with phlegm, fever, chest pain and shortness of breath. It is diagnosed by examination and chest X-ray and often needs antibiotics. Seek care promptly."},
    {"topic": "Migraine", "text": "Migraine causes throbbing, often one-sided headache with nausea and sensitivity to light. Rest in a dark quiet room and drink water. Recurrent migraines should be reviewed by a neurologist."},
    {"topic": "Dengue", "text": "Dengue causes high fever, severe headache, pain behind the eyes, joint pain and rash. Drink plenty of fluids, avoid ibuprofen and aspirin, and seek urgent care for bleeding, repeated vomiting or severe abdominal pain."},
    {"topic": "Diabetes", "text": "Symptoms include excessive thirst, frequent urination, tiredness and blurred vision. Diagnosis is by blood glucose and HbA1c tests. Management includes diet, exercise and medication as prescribed by a doctor."},
    {"topic": "High blood pressure", "text": "High blood pressure often has no symptoms. Reducing salt, exercising, keeping a healthy weight and taking prescribed medication help. Check blood pressure regularly."},
    {"topic": "Skin cancer warning signs", "text": "Check moles using ABCDE: Asymmetry, irregular Border, uneven Color, Diameter over 6 mm, and Evolving change. Any mole with these signs should be examined by a dermatologist."},
    {"topic": "Skin protection", "text": "Use broad-spectrum sunscreen SPF 30 or higher, avoid midday sun, wear protective clothing and avoid tanning beds."},
    {"topic": "Dehydration", "text": "Signs are thirst, dark urine, dizziness and dry mouth. Drink fluids or oral rehydration solution. Seek care for confusion, fainting or inability to keep fluids down."},
]

EMERGENCY_TERMS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "unconscious", "stroke", "seizure", "severe bleeding", "suicide",
    "overdose", "heart attack", "poison",
]

DISCLAIMER = ("_This is general information, not a diagnosis. "
              "Please consult a qualified healthcare professional._")

_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_matrix = _vectorizer.fit_transform(
    [d["topic"] + " " + d["text"] for d in KNOWLEDGE_BASE]
)


def retrieve(question, k=2, min_score=0.08, relative=0.6):
    scores = cosine_similarity(_vectorizer.transform([question]), _matrix)[0]
    best = scores.argsort()[::-1][:k]
    if scores[best[0]] < min_score:
        return []
    top = scores[best[0]]
    return [(KNOWLEDGE_BASE[i], float(scores[i]))
            for i in best if scores[i] >= min_score and scores[i] >= relative * top]


def chat(question):
    q = question.lower()
    if any(term in q for term in EMERGENCY_TERMS):
        return ("⚠️ This could be an emergency. Call your local emergency number "
                "(112 in India) or go to the nearest hospital immediately.")
    hits = retrieve(question)
    if not hits:
        return ("I don't have reliable information on that. "
                "Please consult a doctor.\n\n" + DISCLAIMER)
    answer = "\n\n".join(f"**{d['topic']}**: {d['text']}" for d, _ in hits)
    return answer + "\n\n" + DISCLAIMER