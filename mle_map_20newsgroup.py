import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# 1. Load Dataset
# -------------------------------
categories = ['sci.space', 'rec.sport.baseball']  # binary subset for simplicity

data = fetch_20newsgroups(subset='train',
                         categories=categories,
                         remove=('headers', 'footers', 'quotes'))

texts = data.data

# -------------------------------
# 2. Text Vectorization
# -------------------------------
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts)

# Convert to numpy array (word counts)
X = X.toarray()

# -------------------------------
# 3. Aggregate Word Counts
# -------------------------------
word_counts = np.sum(X, axis=0)  # total counts per word
total_count = np.sum(word_counts)

# -------------------------------
# 4. MLE Estimation
# -------------------------------
theta_mle = word_counts / total_count

# -------------------------------
# 5. MAP Estimation (Dirichlet Prior)
# -------------------------------

def compute_map(alpha):
    alpha_vec = np.ones_like(word_counts) * alpha
    theta_map = (word_counts + alpha_vec - 1) / np.sum(word_counts + alpha_vec - 1)
    return theta_map

# Different priors
theta_map_1 = compute_map(alpha=1)   # Uniform prior
theta_map_2 = compute_map(alpha=2)   # Mild smoothing
theta_map_10 = compute_map(alpha=10) # Strong smoothing

# -------------------------------
# 6. Compare Top Words
# -------------------------------
feature_names = np.array(vectorizer.get_feature_names_out())

def top_words(theta, name):
    top_idx = np.argsort(theta)[-10:]
    print(f"\nTop words ({name}):")
    print(feature_names[top_idx])

top_words(theta_mle, "MLE")
top_words(theta_map_1, "MAP alpha=1")
top_words(theta_map_2, "MAP alpha=2")
top_words(theta_map_10, "MAP alpha=10")

# -------------------------------
# 7. Compare Probability Spread
# -------------------------------
print("\n=== Probability Statistics ===")
print("MLE min prob:", np.min(theta_mle))
print("MAP (alpha=1) min prob:", np.min(theta_map_1))
print("MAP (alpha=10) min prob:", np.min(theta_map_10))

# -------------------------------
# 8. Visualization
# -------------------------------
import matplotlib.pyplot as plt

plt.figure()

# Sort probabilities for visualization
plt.plot(np.sort(theta_mle), label="MLE")
plt.plot(np.sort(theta_map_2), label="MAP alpha=2")
plt.plot(np.sort(theta_map_10), label="MAP alpha=10")

plt.xlabel("Word Index (sorted)")
plt.ylabel("Probability")
plt.title("MLE vs MAP (Dirichlet Prior)")
plt.legend()
plt.grid()

plt.show()