# recognition/evaluate.py

class Evaluator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def evaluate(self):
        results = self.model.evaluate(self.test_data)
        print(f"Test Accuracy: {results[1] * 100:.2f}%")
