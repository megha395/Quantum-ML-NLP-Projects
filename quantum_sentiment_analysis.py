# %%
import pandas as pd
import re
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormalizerSequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np

# %% [markdown]
# Step 1: Load and Clean Tweet Data

# %%
df = pd.read_csv("train_data/train.csv")  # Replace with your actual CSV path
df = df.dropna(subset=["text"])

texts = df["text"].astype(str).tolist()

# %%
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

cleaned_texts = [clean_tweet(t) for t in texts]
df["cleaned_text"] = cleaned_texts

# Save cleaned tweets to a file for BPE training
with open("cleaned_tweets.txt", "w", encoding="utf-8") as f:
    for line in cleaned_texts:
        f.write(line + "\n")

# %%
df

# %% [markdown]
# GPT2 Tokenizer and Embedding

# %% [markdown]
# check for rows with only white spaces or is empty

# %%
# Identify rows where cleaned_text is empty or only whitespace
empty_or_whitespace_rows = df[df["cleaned_text"].str.strip() == ""]

# Display them
print("Rows with empty or whitespace-only 'cleaned_text':")
print(empty_or_whitespace_rows)


# %%
df = df[df["cleaned_text"].str.strip().astype(bool)]  # Remove empty or whitespace-only rows

# %%
df

# %%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

def get_gpt2_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=50)
    # Decode the token ids into readable tokens
    token_ids = inputs['input_ids'][0]
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]

    print(f"Original text: {text}")
    print(f"Token IDs: {token_ids.tolist()}")
    print(f"Tokens: {tokens}")
    print("-" * 50)

    with torch.no_grad():
        outputs = model(**inputs)
        
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Generating GPT-2 embeddings...")
df["embedding"] = df["cleaned_text"].apply(get_gpt2_embedding)
X_embed = np.stack(df["embedding"].values)


# %%
df

# %% [markdown]
# Label 

# %%
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])
y = to_categorical(y, num_classes=3)

# %% [markdown]
# Classical Model

# %%
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_embed, y, test_size=0.2, random_state=42)

# %%
start_cls = time.time()
model_classical = Sequential([
    Dense(128, activation='relu', input_shape=(X_embed.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
model_classical.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_classical.fit(X_train_c, y_train_c, validation_data=(X_test_c, y_test_c), epochs=50, batch_size=32)
loss_c, acc_c = model_classical.evaluate(X_test_c, y_test_c)
print(f"\n Classical Model Accuracy: {acc_c:.4f}")
end_cls = time.time()

# %% [markdown]
# Comparison of Quantum Angle encoding vs Amplitude Encoding

# %% [markdown]
# Quantum Feature Map (ZZFeatureMap)

# %%
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# %% [markdown]
# PCA Variance Explained Plot

# %%
pca_all = PCA(n_components=50)
X_pca_all = pca_all.fit_transform(X_embed)
explained = np.cumsum(pca_all.explained_variance_ratio_)
plt.figure(figsize=(8, 4))
plt.plot(range(1, 51), explained, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
pca_zz = PCA(n_components=4)
X_reduced_zz = pca_zz.fit_transform(X_embed)

# %%
X_reduced_zz

# %%
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)

def quantum_map(vec):
    param_dict = dict(zip(feature_map.parameters, vec))
    circuit = feature_map.assign_parameters(param_dict)
    state = Statevector.from_instruction(circuit)
    return np.abs(state.data[:4])**2

print("Generating quantum-mapped vectors (ZZFeatureMap)...")
X_quantum_zz = np.array([quantum_map(vec) for vec in X_reduced_zz])
X_quantum_zz = np.expand_dims(X_quantum_zz, axis=1)

# %% [markdown]
# Amplitude Encoding

# %%
pca_amp = PCA(n_components=4)
X_reduced_amp = pca_amp.fit_transform(X_embed)
X_normalized_amp = normalize(X_reduced_amp, norm='l2')

# Padding to 8-dim for simulation compatibility with LSTM input shape (1, 8)
X_padded_amp = np.pad(X_normalized_amp, ((0, 0), (0, 4)), mode='constant')

def amplitude_encode(vec):
    state = Statevector(vec)
    return np.abs(state.data)  # Already normalized

print("Generating amplitude-encoded vectors...")
X_amplitude = np.array([amplitude_encode(x) for x in X_padded_amp])
X_amplitude = np.expand_dims(X_amplitude, axis=1)


# %%
X_amplitude


# %% [markdown]
# Train & Evaluate Both Models

# %%
X_train_zz, X_test_zz, y_train_zz, y_test_zz = train_test_split(X_quantum_zz, y, test_size=0.2, random_state=42)
X_train_amp, X_test_amp, y_train_amp, y_test_amp = train_test_split(X_amplitude, y, test_size=0.2, random_state=42)

# %%
start_zz = time.time()
model_zz = Sequential([
    LSTM(128, input_shape=(1, 4)),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])
model_zz.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Training model on ZZFeatureMap quantum features...")
model_zz.fit(X_train_zz, y_train_zz, validation_data=(X_test_zz, y_test_zz), epochs=50, batch_size=32)
loss_zz, acc_zz = model_zz.evaluate(X_test_zz, y_test_zz)
end_zz = time.time()

# %%
start_amp = time.time()
model_amp = Sequential([
    LSTM(128, input_shape=(1, 8)),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])
model_amp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Training model on amplitude-encoded quantum features...")
model_amp.fit(X_train_amp, y_train_amp, validation_data=(X_test_amp, y_test_amp), epochs=50, batch_size=32)
loss_amp, acc_amp = model_amp.evaluate(X_test_amp, y_test_amp)
end_amp = time.time()

# %% [markdown]
# Hybrid Model (GPT-2 + ZZFeatureMap)

# %%
X_zz_hybrid = np.concatenate([X_embed, X_quantum_zz.squeeze()], axis=1)
X_zz_hybrid_seq = np.expand_dims(X_zz_hybrid, axis=1)
X_train_hyb, X_test_hyb, y_train_hyb, y_test_hyb = train_test_split(X_zz_hybrid_seq, y, test_size=0.2, random_state=42)

start_hyb = time.time()
model_hyb = Sequential([
    LSTM(128, input_shape=(1, X_zz_hybrid.shape[1])),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])
model_hyb.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Training hybrid LSTM model on GPT-2 + ZZFeatureMap features...")
model_hyb.fit(X_train_hyb, y_train_hyb, validation_data=(X_test_hyb, y_test_hyb), epochs=50, batch_size=32)
loss_hyb, acc_hyb = model_hyb.evaluate(X_test_hyb, y_test_hyb)
end_hyb = time.time()


# %%
print(f" Accuracy with ZZFeatureMap: {acc_zz:.4f} | Time: {end_zz - start_zz:.2f}s")
print(f" Accuracy with Amplitude Encoding: {acc_amp:.4f} | Time: {end_amp - start_amp:.2f}s")
print(f" Accuracy with Classical LSTM: {acc_c:.4f} | Time: {end_cls - start_cls:.2f}s")
print(f" Accuracy with Hybrid Model: {acc_hyb:.4f} | Time: {end_hyb - start_hyb:.2f}s")

# %% [markdown]
# Bar Chart for Accuracy & Runtime

# %%
labels = ['ZZFeatureMap', 'Amplitude', 'Classical', 'Hybrid']
accuracies = [acc_zz, acc_amp, acc_c, acc_hyb]
times = [end_zz - start_zz, end_amp - start_amp, end_cls - start_cls, end_hyb - start_hyb]

x = np.arange(len(labels))  # label locations
width = 0.35  # width of bars

fig, ax1 = plt.subplots(figsize=(10, 5))

bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, times, width, label='Runtime (s)', color='salmon')

ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Runtime (s)')
ax1.set_ylim(0, 1.1)
ax2.set_ylim(0, max(times)*1.2)
ax1.set_title('Comparison: Classical, Quantum & Hybrid Models')

# X-axis labels
plt.xticks(x, labels, rotation=20)

# Legends
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

fig.tight_layout()
plt.show()



