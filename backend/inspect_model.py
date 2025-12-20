import tensorflow.lite as tflite
import os

model_path = "backend/models/model.tflite"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== Input Details ===")
for i, detail in enumerate(input_details):
    print(f"Input {i}: {detail['name']}, Shape: {detail['shape']}, Index: {detail['index']}")

print("\n=== Output Details ===")
for i, detail in enumerate(output_details):
    print(f"Output {i}: {detail['name']}, Shape: {detail['shape']}, Index: {detail['index']}")
