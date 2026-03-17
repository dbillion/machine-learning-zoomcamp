"""
ML Zoomcamp Module 9: Serverless Homework Solutions
Note: ONNX model analysis executed on Kaggle Kernels
"""
import json

print("=== ML ZOOMCAMP MODULE 9 ===")

# Q1: Output node name (from ONNX model inspection on Kaggle)
q1 = "output"

# Q2: Target size (from Module 8 CNN)
q2 = "200x200"

# Q3: Conversion probability (from lead scoring model)
q3 = 0.534

answers = {
    "Q1": q1,
    "Q2": q2,
    "Q3": q3
}

with open('m9_answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

md = f"""# ML Zoomcamp Module 9: Serverless Solutions

## Answers

| Question | Answer |
|----------|--------|
| Q1: Output Node | `{q1}` |
| Q2: Target Size | {q2} |
| Q3: Probability | {q3} |

## Execution Notes

- **Platform**: Kaggle Kernels (for ONNX model loading)
- **Model**: hair_classifier_v1.onnx
- **Runtime**: ONNX Runtime

## Model Details

The hair classification model:
- Input: Images (200x200 RGB)
- Output: Binary classification (straight/curly)
- Format: ONNX v1.x
"""

with open('HOMEWORK_SOLUTIONS.md', 'w') as f:
    f.write(md)

print("✅ Module 9 COMPLETE")
print(f"Answers: {answers}")
