from recognition.dataset import load_data
from recognition.model import create_model
from tensorflow.keras.callbacks import EarlyStopping

def train_model():
    X_train, y_train, X_test, y_test = load_data()

    model = create_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    model.save('ocr_model.h5')

if __name__ == "__main__":
    train_model()