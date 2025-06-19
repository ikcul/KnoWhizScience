# KnoWhiz Prompt & Flashcard Generator

This repository showcases my work on KnoWhiz, a local AI-powered educational pipeline that generates structured study materials (flashcards, quizzes, tests) from zero-shot or textbook-based prompts.

## 💡 What It Does

- Accepts **natural language course prompts** across topics like Math, Computer Science, and Physics
- Uses LLMs to auto-generate:
  - 📘 Chapter breakdowns
  - 🧠 Keyword-rich flashcards
  - ❓ Quiz and test questions (MCQs & short answer)

## 🧪 Technologies Used

- Python 3.11
- OpenAI & LangChain APIs
- Multiprocessing for batch runs
- Structured output in JSON for interoperability
- VS Code & Git for local development

## 🔍 Example Use Case

Prompt:
level:"Intermediate", subject:"Mathematics", text:"Understand vector calculus and theorems like Stokes and Gauss"


➡️ Outputs:
- `flashcards_set*.json`: Concept breakdowns + examples
- `quiz/*.json`: MCQs based on generated content
- `test/*.json`: Short-answer and free-response questions

## 🚀 How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Add your `.env` with API keys:
OPENAI_API_KEY=your_key

3. Run locally:
python local_test.py

---

## 👨‍💻 Author

**Daniel Sim**  
Undergraduate focused on AI, Mathematics, and Scientific Computing  
*Check out more at:* [github.com/ikcul](https://github.com/ikcul)

---

## 📝 License

This project is for educational demonstration purposes.

