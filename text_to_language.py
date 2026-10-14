import csv
import json
import os
import time

from collections import defaultdict

import numpy as np


# ============================================================
# Configuration
# ============================================================

MAX_LENGTH = 160
EMBEDDING_DIM = 24
HIDDEN_DIM = 48

LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 64

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ============================================================
# Custom recurrent neural network
# ============================================================

class CharacterRNN:

    def __init__(
        self,
        vocab_size,
        num_languages,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        max_length=MAX_LENGTH,
        seed=42
    ):
        self.vocab_size = vocab_size
        self.num_languages = num_languages
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        rng = np.random.default_rng(seed)

        # Character embedding matrix.
        self.E = rng.normal(
            0.0,
            0.05,
            (vocab_size, embedding_dim)
        )

        # Input-to-hidden weights.
        self.Wx = rng.normal(
            0.0,
            np.sqrt(1.0 / embedding_dim),
            (embedding_dim, hidden_dim)
        )

        # Hidden-to-hidden recurrent weights.
        self.Wh = rng.normal(
            0.0,
            np.sqrt(1.0 / hidden_dim),
            (hidden_dim, hidden_dim)
        )

        self.bh = np.zeros(hidden_dim)

        # Hidden-to-output weights.
        self.Wo = rng.normal(
            0.0,
            np.sqrt(1.0 / hidden_dim),
            (hidden_dim, num_languages)
        )

        self.bo = np.zeros(num_languages)

        self.parameters = {
            "E": self.E,
            "Wx": self.Wx,
            "Wh": self.Wh,
            "bh": self.bh,
            "Wo": self.Wo,
            "bo": self.bo
        }

        # Adam optimizer state.
        self.adam_m = {
            name: np.zeros_like(param)
            for name, param in self.parameters.items()
        }

        self.adam_v = {
            name: np.zeros_like(param)
            for name, param in self.parameters.items()
        }

        self.adam_step = 0

    # ========================================================
    # Forward propagation
    # ========================================================

    def forward(self, x, cache=False):
        """
        x: integer array with shape (batch_size, sequence_length)

        Returns:
            probabilities: (batch_size, num_languages)
            cache: intermediate values for backpropagation
        """

        batch_size, sequence_length = x.shape

        # Initial hidden state.
        h = np.zeros(
            (batch_size, self.hidden_dim),
            dtype=np.float64
        )

        previous_states = []
        candidate_states = []
        masks = []

        for t in range(sequence_length):

            previous_h = h

            # Embedding lookup.
            embedded = self.E[x[:, t]]

            # Padding positions must not change the hidden state.
            mask = (
                x[:, t] != 0
            ).astype(np.float64)[:, None]

            # Recurrent transformation.
            z = (
                embedded @ self.Wx
                + previous_h @ self.Wh
                + self.bh
            )

            candidate = np.tanh(z)

            # Keep the previous state when processing padding.
            h = (
                candidate * mask
                + previous_h * (1.0 - mask)
            )

            if cache:
                previous_states.append(previous_h)
                candidate_states.append(candidate)
                masks.append(mask)

        # Final hidden state -> language scores.
        logits = h @ self.Wo + self.bo

        # Numerically stable softmax.
        shifted_logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )

        exp_logits = np.exp(shifted_logits)

        probabilities = exp_logits / np.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )

        if not cache:
            return probabilities

        memory = {
            "x": x,
            "last_hidden": h,
            "probabilities": probabilities,
            "previous_states": previous_states,
            "candidate_states": candidate_states,
            "masks": masks
        }

        return probabilities, memory

    # ========================================================
    # Cross-entropy loss
    # ========================================================

    @staticmethod
    def cross_entropy(probabilities, labels):
        selected = probabilities[
            np.arange(len(labels)),
            labels
        ]

        return float(
            -np.mean(np.log(selected + 1e-12))
        )

    # ========================================================
    # Backpropagation through time
    # ========================================================

    def backward(self, labels, memory):
        x = memory["x"]
        probabilities = memory["probabilities"]
        last_hidden = memory["last_hidden"]

        batch_size, sequence_length = x.shape

        gradients = {
            name: np.zeros_like(param)
            for name, param in self.parameters.items()
        }

        # Gradient of cross-entropy with softmax.
        d_logits = probabilities.copy()

        d_logits[
            np.arange(batch_size),
            labels
        ] -= 1.0

        d_logits /= batch_size

        # Output layer gradients.
        gradients["Wo"] = last_hidden.T @ d_logits
        gradients["bo"] = np.sum(d_logits, axis=0)

        # Propagate gradients into the recurrent layer.
        dh = d_logits @ self.Wo.T

        for t in reversed(range(sequence_length)):

            previous_h = memory["previous_states"][t]
            candidate = memory["candidate_states"][t]
            mask = memory["masks"][t]

            # Split gradients between the candidate state
            # and the unchanged state used for padding.
            d_candidate = dh * mask

            d_previous_direct = dh * (1.0 - mask)

            # Derivative of tanh.
            dz = d_candidate * (1.0 - candidate ** 2)

            # Recurrent parameter gradients.
            embedded = self.E[x[:, t]]

            gradients["Wx"] += embedded.T @ dz
            gradients["Wh"] += previous_h.T @ dz
            gradients["bh"] += np.sum(dz, axis=0)

            # Backpropagate into character embeddings.
            d_embedded = dz @ self.Wx.T

            np.add.at(
                gradients["E"],
                x[:, t],
                d_embedded
            )

            # Continue backwards through the recurrent state.
            dh = (
                dz @ self.Wh.T
                + d_previous_direct
            )

        return gradients

    # ========================================================
    # Adam optimizer with gradient clipping
    # ========================================================

    def update(self, gradients, learning_rate=LEARNING_RATE):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        # Global gradient clipping.
        global_norm = np.sqrt(
            sum(
                np.sum(gradient ** 2)
                for gradient in gradients.values()
            )
        )

        if global_norm > 5.0:
            scale = 5.0 / (global_norm + 1e-12)

            gradients = {
                name: gradient * scale
                for name, gradient in gradients.items()
            }

        self.adam_step += 1

        for name, parameter in self.parameters.items():

            gradient = gradients[name]

            self.adam_m[name] = (
                beta1 * self.adam_m[name]
                + (1.0 - beta1) * gradient
            )

            self.adam_v[name] = (
                beta2 * self.adam_v[name]
                + (1.0 - beta2) * gradient ** 2
            )

            m_hat = self.adam_m[name] / (
                1.0 - beta1 ** self.adam_step
            )

            v_hat = self.adam_v[name] / (
                1.0 - beta2 ** self.adam_step
            )

            parameter -= learning_rate * m_hat / (
                np.sqrt(v_hat) + epsilon
            )

    # ========================================================
    # Train one mini-batch
    # ========================================================

    def train_batch(self, x, labels):
        probabilities, memory = self.forward(
            x,
            cache=True
        )

        loss = self.cross_entropy(
            probabilities,
            labels
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        accuracy = np.mean(predictions == labels)

        gradients = self.backward(
            labels,
            memory
        )

        self.update(gradients)

        return loss, float(accuracy)

    # ========================================================
    # Evaluate a dataset
    # ========================================================

    def evaluate(self, x, labels, batch_size=128):
        total_loss = 0.0
        correct = 0
        total = len(labels)

        for start in range(0, total, batch_size):

            end = min(start + batch_size, total)

            batch_x = x[start:end]
            batch_y = labels[start:end]

            probabilities = self.forward(batch_x)

            loss = self.cross_entropy(
                probabilities,
                batch_y
            )

            batch_count = len(batch_y)

            total_loss += loss * batch_count

            predictions = np.argmax(
                probabilities,
                axis=1
            )

            correct += int(
                np.sum(predictions == batch_y)
            )

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1)
        }


# ============================================================
# Text preprocessing
# ============================================================

def encode_text(text, char_to_id, max_length=MAX_LENGTH):
    text = text.casefold()

    ids = [
        char_to_id.get(
            character,
            char_to_id[UNK_TOKEN]
        )
        for character in text[:max_length]
    ]

    ids += [char_to_id[PAD_TOKEN]] * (
        max_length - len(ids)
    )

    return ids


def load_csv_dataset(path):
    texts = []
    labels = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames or not {
            "text",
            "language"
        }.issubset(reader.fieldnames):

            raise ValueError(
                "CSV must contain 'text' and 'language' columns."
            )

        for row in reader:

            text = (row.get("text") or "").strip()
            language = (row.get("language") or "").strip()

            if text and language:
                texts.append(text)
                labels.append(language)

    if len(texts) < 10:
        raise ValueError(
            "At least 10 valid text examples are required."
        )

    return texts, labels


def prepare_dataset(texts, labels, seed=42):
    """
    Split each language separately into training and validation
    sets so that every language is represented in both.
    """

    grouped = defaultdict(list)

    for text, language in zip(texts, labels):
        grouped[language].append(text)

    rng = np.random.default_rng(seed)

    train_texts = []
    train_labels = []

    val_texts = []
    val_labels = []

    for language, examples in grouped.items():

        if len(examples) < 2:
            raise ValueError(
                f"Language '{language}' requires at least 2 examples."
            )

        indices = rng.permutation(len(examples))

        split = int(len(examples) * 0.8)
        split = max(1, min(split, len(examples) - 1))

        for index in indices[:split]:
            train_texts.append(examples[index])
            train_labels.append(language)

        for index in indices[split:]:
            val_texts.append(examples[index])
            val_labels.append(language)

    return (
        train_texts,
        train_labels,
        val_texts,
        val_labels
    )


def build_vocabulary(texts):
    characters = set()

    for text in texts:
        characters.update(text.casefold())

    char_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for character in sorted(characters):
        if character not in char_to_id:
            char_to_id[character] = len(char_to_id)

    return char_to_id


def encode_dataset(texts, labels, char_to_id, label_to_id):
    x = np.asarray(
        [
            encode_text(text, char_to_id)
            for text in texts
        ],
        dtype=np.int32
    )

    y = np.asarray(
        [
            label_to_id[label]
            for label in labels
        ],
        dtype=np.int64
    )

    return x, y


# ============================================================
# Model saving and loading
# ============================================================

def save_model(model, char_to_id, languages, path):
    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True
    )

    metadata = {
        "char_to_id": char_to_id,
        "languages": languages,
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim,
        "max_length": model.max_length
    }

    np.savez_compressed(
        path,
        metadata=np.array(json.dumps(metadata)),
        E=model.E,
        Wx=model.Wx,
        Wh=model.Wh,
        bh=model.bh,
        Wo=model.Wo,
        bo=model.bo
    )


def load_language_model(path):
    if not os.path.exists(path):
        return None

    with np.load(path, allow_pickle=False) as data:

        metadata = json.loads(
            str(data["metadata"].item())
        )

        char_to_id = metadata["char_to_id"]
        languages = metadata["languages"]

        model = CharacterRNN(
            vocab_size=len(char_to_id),
            num_languages=len(languages),
            embedding_dim=metadata["embedding_dim"],
            hidden_dim=metadata["hidden_dim"],
            max_length=metadata["max_length"]
        )

        for name in model.parameters:
            model.parameters[name][...] = data[name]

    return NumpyLanguageDetector(
        model,
        char_to_id,
        languages
    )


# ============================================================
# Prediction interface
# ============================================================

class NumpyLanguageDetector:

    def __init__(self, model, char_to_id, languages):
        self.model = model
        self.char_to_id = char_to_id
        self.languages = languages

    def predict(self, text, top_k=3):
        text = text.strip()

        if not text or not any(c.isalpha() for c in text):
            return {
                "language": "Unknown",
                "confidence": 0.0,
                "alternatives": [],
                "inference_time_ms": 0.0
            }

        x = np.asarray(
            [
                encode_text(
                    text,
                    self.char_to_id,
                    self.model.max_length
                )
            ],
            dtype=np.int32
        )

        start = time.perf_counter()

        probabilities = self.model.forward(x)[0]

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        indices = np.argsort(probabilities)[::-1]

        indices = indices[:min(top_k, len(self.languages))]

        ranked = [
            {
                "language": self.languages[index],
                "confidence": float(probabilities[index])
            }
            for index in indices
        ]

        return {
            "language": ranked[0]["language"],
            "confidence": ranked[0]["confidence"],
            "alternatives": ranked[1:],
            "inference_time_ms": elapsed_ms
        }


# ============================================================
# Training
# ============================================================

def train_language_model(
    csv_path,
    model_path,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    progress_callback=None,
    stop_event=None
):
    started = time.perf_counter()

    def report(stats):
        if progress_callback:
            progress_callback(stats)

    # Load and split the data.
    texts, raw_labels = load_csv_dataset(csv_path)

    languages = sorted(set(raw_labels))

    train_texts, train_labels, val_texts, val_labels = (
        prepare_dataset(texts, raw_labels)
    )

    char_to_id = build_vocabulary(train_texts)

    label_to_id = {
        language: index
        for index, language in enumerate(languages)
    }

    x_train, y_train = encode_dataset(
        train_texts,
        train_labels,
        char_to_id,
        label_to_id
    )

    x_val, y_val = encode_dataset(
        val_texts,
        val_labels,
        char_to_id,
        label_to_id
    )

    # Initialize the network.
    model = CharacterRNN(
        vocab_size=len(char_to_id),
        num_languages=len(languages)
    )

    best_val_loss = float("inf")
    best_epoch = 0

    rng = np.random.default_rng(42)

    for epoch in range(epochs):

        if stop_event is not None and stop_event.is_set():
            break

        # Shuffle the training set.
        order = rng.permutation(len(x_train))

        for start in range(0, len(order), batch_size):

            if stop_event is not None and stop_event.is_set():
                break

            indices = order[start:start + batch_size]

            model.train_batch(
                x_train[indices],
                y_train[indices]
            )

        if stop_event is not None and stop_event.is_set():
            break

        # Evaluate after each epoch.
        train_stats = model.evaluate(
            x_train,
            y_train
        )

        val_stats = model.evaluate(
            x_val,
            y_val
        )

        if val_stats["loss"] < best_val_loss:

            best_val_loss = val_stats["loss"]
            best_epoch = epoch + 1

            save_model(
                model,
                char_to_id,
                languages,
                model_path
            )

        report({
            "epoch": epoch + 1,
            "epochs": epochs,

            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],

            "train_accuracy": train_stats["accuracy"],
            "val_accuracy": val_stats["accuracy"],

            "samples": len(x_train),
            "validation_samples": len(x_val),
            "languages": len(languages),

            "device": "CPU (NumPy)",

            "elapsed_seconds": (
                time.perf_counter() - started
            ),

            "best_epoch": best_epoch
        })

    if not os.path.exists(model_path):
        raise RuntimeError(
            "No model was saved. Check the dataset or training settings."
        )

    # Return the best saved model, rather than the last epoch.
    return load_language_model(model_path)