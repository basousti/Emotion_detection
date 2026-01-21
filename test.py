import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Simple text samples
texts = [
    "I am very happy today",
    "This is the best day ever",
    "I feel sad and lonely",
    "I am depressed",
    "I am angry at you",
    "This makes me furious",
]

# Emotion labels
labels = [
    "joy",
    "joy",
    "sadness",
    "sadness",
    "anger",
    "anger"
]

encoder = LabelEncoder() 
y = encoder.fit_transform(labels)

print("Emotion classes:", encoder.classes_)
print("Encoded labels:", y) 

 
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post') 

print("Example sequence:", sequences)
print("padded sequences :", padded_sequences)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    padded_sequences,
    y,
    epochs=30,
    verbose=1
)
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=padded_sequences.shape[1], padding='post')
    prediction = model.predict(pad)
    emotion = encoder.inverse_transform([prediction.argmax()])
    return emotion[0]



print(predict_emotion("I feel very happy today"))
print(predict_emotion("I am sad and tired"))
print(predict_emotion("I am extremely angry"))
print(predict_emotion("I'm excited"))

print("What if it works ")
