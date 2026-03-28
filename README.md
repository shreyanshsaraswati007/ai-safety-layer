🔒 OpenAI Safety Layer: Real-Time LLM Defense & Evaluation Platform
A comprehensive AI safety framework that combines rule-based detection with machine learning to identify, classify, and defend against jailbreak attempts and harmful prompts in Large Language Model interactions.
🌟 Features
•  Hybrid Detection Engine: Combines rule-based keyword detection with ML-based semantic analysis
•  Multi-Attack Classification: Detects 4 types of jailbreak attempts:
•  instruction_override — Attempts to override system instructions
•  role_play — Character/persona-based manipulation
•  multi_step — Multi-turn conversational traps
•  illegal_activity — Direct requests for harmful/illegal content
•  Adaptive Defense Mechanisms: Context-aware responses (block, rephrase, or flag)
•  Real-time Web Interface: Interactive Gradio UI for testing and demonstration
•  Comprehensive Evaluation: Built-in metrics, confusion matrices, and performance visualizations
🏗️ Architecture
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Prompt   │────▶│  SafetyDetector │────▶│  Defense Engine │
│                 │     │   (Hybrid ML)     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
│
┌─────────┴─────────┐
▼                   ▼
┌─────────────┐       ┌─────────────┐
│ Rule-Based  │       │  ML-Based   │
│  Detection  │       │  (BERT+LR)  │
│  (Keywords) │       │  (Embeddings)│
└─────────────┘       └─────────────┘
🚀 Quick Start
Prerequisites
pip install pandas numpy scikit-learn sentence-transformers matplotlib seaborn gradio
Running the Platform
python ai_safety_layer.py
This will:
1.  Generate the synthetic safety dataset (62 examples)
2.  Train the hybrid detection model
3.  Display evaluation metrics and visualizations
4.  Launch the Gradio web interface at http://127.0.0.1:7860
📊 Detection Methodology
Rule-Based Detection
•  Keyword matching against curated attack pattern dictionaries
•  Weighted risk scoring (15-30 points per match based on severity)
•  Attack type classification with 100-point risk scale cap
ML-Based Detection
•  Sentence embeddings via all-MiniLM-L6-v2 transformer
•  Logistic Regression classifier for jailbreak probability
•  Probability threshold tuning for precision/recall balance
Hybrid Scoring Formula
hybrid_risk_score = (normalized_rule_risk × rule_weight × 100) +
(ml_probability × ml_weight × 100)
Default weights: ml_weight=0.6, rule_weight=0.4
🛡️ Defense Strategies
Risk Score	Attack Type	Action
≥60 + illegal_activity/instruction_override	BLOCK	Hard refusal with safety message
≥60 + role_play/multi_step	REPHRASE	Neutralized response with context
≥60 (general)	FLAG	Warning with ethical guidelines
<60	ALLOW	Pass through unchanged
📈 Performance Metrics
The system achieves strong detection performance on the synthetic dataset:
Metric	Score
Accuracy	1.00
Precision	1.00
Recall	1.00
F1-Score	1.00
Note: Performance evaluated on 62-example synthetic dataset with 70/30 train-test split.
🖥️ Web Interface
Access the interactive demo at http://127.0.0.1:7860 to:
•  Input custom prompts for real-time analysis
•  View hybrid risk scores and detected attack types
•  See defense outputs and explanations
•  Test edge cases and adversarial examples
📁 Project Structure
ai_safety_layer.py          # Main application file
├── create_safety_dataset()  # Synthetic data generation (62 examples)
├── SafetyDetector           # Core detection class
│   ├── rule_based_detect()  # Keyword-based detection
│   ├── train_ml_detector()  # ML model training
│   ├── ml_predict()         # Embedding-based prediction
│   └── hybrid_detect()      # Combined scoring
├── defend_prompt()          # Defense mechanism router
├── evaluate_detector()      # Metrics & validation
├── plot_*()                 # Visualization functions
└── create_gradio_interface() # Web UI
🔧 Customization
Adding New Attack Patterns
detector.rule_based_keywords["new_category"] = [
"pattern1", "pattern2", "pattern3"
]
Adjusting Detection Sensitivity
Increase ML influence for semantic understanding
results = detector.hybrid_detect(prompt, ml_weight=0.8, rule_weight=0.2)
Lower threshold for more aggressive blocking
defense = defend_prompt(prompt, detector, threshold=50)
Training on Custom Data
detector = SafetyDetector()
detector.train_ml_detector(custom_prompts, custom_labels)
⚠️ Limitations & Considerations
•  Synthetic Dataset: Current implementation uses 62 synthetic examples; production use requires larger, diverse real-world datasets
•  English Only: Keyword detection optimized for English prompts
•  Evasion Potential: Determined adversaries may craft prompts that bypass keyword lists
•  No Active Learning: Model does not automatically update from new attack patterns
🔮 Future Enhancements
•  [ ] Integration with live LLM APIs (OpenAI, Anthropic, etc.)
•  [ ] Active learning pipeline for continuous improvement
•  [ ] Multi-language support
•  [ ] Adversarial training for robustness
•  [ ] Semantic similarity defense for paraphrased attacks
•  [ ] A/B testing framework for defense strategy optimization
📜 License
This project is provided as a research and educational implementation of AI safety techniques. Use responsibly in accordance with AI safety best practices and applicable regulations.
Created for: LLM Safety Research & Educational Demonstrations
Last Updated: March 2026
