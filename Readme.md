# ⚡ Electricity-bill-calculator-using-ANN

A smart web application for calculating monthly electricity bills using Artificial Neural Networks (ANN), built with **FastAPI** and a modern HTML/CSS/JavaScript frontend. This project enables users to estimate their electricity usage and bill based on TNEB tariff rates, providing accurate and efficient results.

---

## 📝 Introduction

**Electricity-bill-calculator-using-ANN** leverages machine learning to predict and calculate electricity bills for users in Tamil Nadu (TNEB tariff). The backend, powered by Python and FastAPI, hosts a trained ANN model, while the frontend delivers an intuitive interface with PWA capabilities. This project demonstrates the synergy between modern web technologies and AI for real-world utility applications.

---

## 🚀 Features

- **ANN-based Bill Prediction:** Uses a trained neural network for accurate bill estimation.
- **FastAPI Backend:** High-performance Python API with CORS support.
- **Responsive Frontend:** Built with HTML, CSS, and JavaScript.
- **PWA Support:** Installable web app with offline caching (Service Worker).
- **Dockerized Backend:** Easy deployment via Docker.
- **TNEB Tariff Support:** Calculates bills based on Tamil Nadu's electricity tariff.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/selvaganesh19/Electricity-bill-calculator-using-ANN.git
cd Electricity-bill-calculator-using-ANN
```

### 2. Backend Setup

#### Using Docker (Recommended)

```bash
cd backend
docker build -t electricity-bill-backend .
docker run -p 7860:7860 electricity-bill-backend
```

#### Or Local Python Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 3. Frontend Setup

Simply open `frontend/index.html` in your browser, or serve it via any static file server.

---

## 📖 Usage

1. **Start the Backend:** Make sure the FastAPI backend is running (see installation above).
2. **Open Frontend:** Visit `frontend/index.html` in your browser or host the frontend at a static server.
3. **Input Details:** Enter your monthly electricity consumption and relevant details.
4. **Get Bill Estimate:** Click "Calculate" to view the estimated bill based on TNEB tariff.

> **PWA Install:** On supported browsers, you can install the app for offline use!

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes.
4. Push your branch (`git push origin feature-name`).
5. Create a Pull Request.

Please ensure your code follows project style and includes tests if applicable.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, TensorFlow/Keras, joblib
- **Frontend:** HTML, CSS, JavaScript, PWA
- **Containerization:** Docker

---

## 📂 Project Structure

```
Electricity-bill-calculator-using-ANN/
│
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── manifest.json
│   ├── sw.js
│   └── icon/
│
└── README.md
```

---

## ✨ Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [TNEB Tariff Reference](https://www.tnebnet.org/)

---

> For any questions or support, feel free to open an issue!

## License
This project is licensed under the **MIT** License.

---
