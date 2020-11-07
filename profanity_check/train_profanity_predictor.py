import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib

dirname = os.path.dirname(__file__)

# Read in data
data = pd.read_csv(os.path.join(dirname, "data/clean_data.csv"))
texts = data["text"].astype(str)
y = data["is_offensive"]

# Vectorize the text
print("Vectoring...")
vectorizer = TfidfVectorizer(stop_words="english", min_df=0.0001)
X = vectorizer.fit_transform(texts)

# Train the model
print("Training...")
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-4, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)

# Save the model
print("Saving...")
joblib.dump(vectorizer, os.path.join(dirname, "data/vectorizer.joblib"))
joblib.dump(cclf, os.path.join(dirname, "data/model.joblib"))

print("Done!")
