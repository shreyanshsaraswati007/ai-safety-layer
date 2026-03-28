🛡️ OpenAI Safety Layer: Real-Time LLM Defense & Evaluation Platform

A real-time AI safety system that detects, classifies, and defends against jailbreak attacks in Large Language Models (LLMs) using a hybrid approach combining rule-based detection and machine learning.

⸻

🚀 Features
	•	Hybrid detection (rule-based + ML)
	•	Risk scoring (0–100)
	•	Attack classification:
	•	instruction_override
	•	role_play
	•	multi_step
	•	illegal_activity
	•	Defense system (block / rephrase / flag / allow)
	•	Interactive web interface (Gradio)
	•	Evaluation metrics (accuracy, precision, recall, F1-score)
	•	Visualization (confusion matrix, performance plots)

⸻

🧠 How It Works

The system processes a user prompt in three stages:
	1.	Detection Engine
	•	Rule-based keyword detection
	•	ML-based semantic detection using embeddings
	2.	Hybrid Scoring
	•	Combines rule-based score and ML probability
	3.	Defense Engine
	•	Blocks, rewrites, flags, or allows the prompt

⸻

⚙️ Detection Method

Rule-Based Detection
	•	Uses predefined keyword patterns
	•	Assigns weighted risk scores (15–30 per match)

ML-Based Detection
	•	Uses sentence embeddings (all-MiniLM-L6-v2)
	•	Logistic Regression classifier
	•	Outputs probability of jailbreak

  Hybrid Formula
  hybrid_risk_score =
(rule_score × 0.4 × 100) +
(ml_probability × 0.6 × 100)

🛡️ Defense Strategy
	•	High risk + illegal/instruction override → BLOCK
	•	High risk + role-play/multi-step → REPHRASE
	•	High risk (general) → FLAG
	•	Low risk → ALLOW

⸻

📊 Performance
	•	Accuracy: 1.00
	•	Precision: 1.00
	•	Recall: 1.00
	•	F1-score: 1.00

Note: Results are based on a synthetic dataset (62 samples).

⸻

🌐 Web Interface

After running the project, a Gradio interface launches where you can:
	•	Enter prompts
	•	View risk scores
	•	See detected attack types
	•	Get defense responses

⸻

🧪 Dataset
	•	62 examples (safe + jailbreak prompts)
	•	Includes multiple attack types:
	•	instruction_override
	•	role_play
	•	multi_step
	•	illegal_activity

How to Run
Install dependencies:

pip install pandas numpy scikit-learn sentence-transformers matplotlib seaborn gradio

Run the project:
python ai_safety_layer.py

Then open:
http://127.0.0.1:7860

 Project Structure
 ai_safety_layer.py

 Includes:
	•	Dataset generation
	•	Detection engine
	•	Defense engine
	•	Evaluation
	•	Visualization
	•	Web interface

🔧 Customization
Add new attack patterns:
detector.rule_based_keywords["new_category"] = ["pattern1", "pattern2"]
Adjust detection weights:
detector.hybrid_detect(prompt, ml_weight=0.8, rule_weight=0.2)
Train on custom data:
detector.train_ml_detector(prompts, labels)

⚠️ Limitations
	•	Uses synthetic dataset
	•	English-only detection
	•	Can be bypassed by advanced prompt engineering
	•	No continuous learning

⸻

🔮 Future Work
	•	Integration with real LLM APIs
	•	Larger real-world datasets
	•	Multi-language support
	•	Adversarial training
	•	Continuous learning pipeline

⸻

👤 Author

Shreyansh Saraswati

⸻

📜 License

This project is for research and educational purposes in AI safety.
Use responsibly.
