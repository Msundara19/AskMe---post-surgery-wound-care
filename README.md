<div align="center">

# 🏥 MediTrack: Real-Time Wound Healing Monitor



**AI-Powered Post-Surgical Care | Real-Time Wound Analysis | LLM-Generated Insights**

click here to view the application : https://askme---post-surgery-wound-care.streamlit.app/ 
[🚀 Live Demo](https://drive.google.com/file/d/1iTxzD--Oofe8pk82E9WOgMAi6oYAU71m/view?usp=drive_link) • [🎯 Features](#-key-features) • [🏗️ Architecture](#️-architecture)

---

</div>

---

## 🎯 The Problem We're Solving

Post-surgical wound care is a critical yet challenging aspect of patient recovery:

| Challenge | Impact | Our Solution |
|-----------|--------|--------------|
| 🚨 **Delayed Intervention** | Complications go unnoticed between appointments | ⚡ Real-time wound monitoring with instant alerts |
| 🏥 **Unnecessary ER Visits** | 30% of ER visits are for normal healing checks | 🤖 AI-powered assessment reduces false alarms |
| 🦠 **Missed Infections** | Early infection signs are hard to spot | 📊 Computer vision detects subtle changes |
| 😰 **Provider Burnout** | Manual follow-up calls consume valuable time | 🔄 Automated tracking with smart alerts |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎯 Real-Time Streaming
- **Live Wound Analysis** using Pathway's streaming engine
- **Sub-second latency** for clinical decision support
- **Automatic metric updates** as new images arrive
- **Trend detection** across multiple observations

</td>
<td width="50%">

### 🧠 Computer Vision Pipeline
- **Adaptive thresholding** for wound region detection
- **Multi-metric extraction**: area, color, redness index
- **Otsu's algorithm** for automatic segmentation
- **OpenCV-based** image preprocessing

</td>
</tr>
<tr>
<td width="50%">

### 🤖 LLM-Powered Insights
- **Groq API** (Llama 3.1-8B) for fast inference
- **Google Gemini** as fallback provider
- **Plain-language summaries** for patients
- **Risk stratification** (low/medium/high)

</td>
<td width="50%">

### 📊 Interactive Dashboard
- **Streamlit** multi-tab interface
- **Plotly** visualizations for trends
- **Historical tracking** of wound metrics
- **Real-time event streaming** display

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        📸 PATIENT UPLOADS IMAGE                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  🧠 COMPUTER VISION PIPELINE (OpenCV)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Preprocessing│→ │  Otsu Thresh │→ │Feature Extract│              │
│  │  RGB + Resize│  │  Segmentation│  │Area, Color, Δ│              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  🤖 LLM ANALYSIS ENGINE (Groq / Google Gemini)                      │
│  • Generates patient-friendly summaries                             │
│  • Risk assessment: Low / Medium / High                             │
│  • Evidence-based recommendations                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ⚡ PATHWAY STREAMING ENGINE (Real-Time Processing)                 │
│  • Watches directory for new analysis results                       │
│  • Streams events to JSONL output                                   │
│  • Powers live dashboard updates                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  📊 STREAMLIT DASHBOARD (Multi-Tab Interface)                       │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                       │
│  │New Scan│ │Progress│ │ Metrics│ │ Stream │                       │
│  │Analysis│ │Tracking│ │ Charts │ │  View  │                       │
│  └────────┘ └────────┘ └────────┘ └────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Category | Technologies | Implementation |
|----------|-------------|----------------|
| 🔥 **Streaming** | Pathway | Real-time data pipeline, watches for new events |
| 🧠 **Computer Vision** | OpenCV | Otsu thresholding, Canny edge detection, color analysis |
| 🤖 **LLM APIs** | Groq (Llama 3.1), Google Gemini | Risk assessment, patient-friendly summaries |
| 🎨 **Frontend** | Streamlit, Plotly | Interactive dashboard with visualizations |
| 🐍 **Language** | Python 3.10+ | Core application logic |

---

## 🚀 Quick Start

### Prerequisites

```bash
✅ Python 3.10+ installed
✅ Git (for cloning)
✅ Groq API key (free tier available)
✅ (Optional) Google Gemini API key
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Msundara19/AskMe---post-surgery-wound-care.git
cd AskMe---post-surgery-wound-care

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run the Application

```bash
# Start Streamlit dashboard
streamlit run streamlit_app_enhanced.py

# (Optional) Start Pathway pipeline in separate terminal
python -m src.meditrack.pipeline.pathway_pipeline
```

---

## 📊 What the System Measures

| Metric | Method | Description |
|--------|--------|-------------|
| **Wound Area** | Pixel counting + Otsu threshold | Estimated area in cm² (calibration-dependent) |
| **Redness Index** | RGB channel analysis | Red-Green difference in wound region |
| **Edge Quality** | Canny edge detection | Boundary sharpness indicator |
| **Healing Score** | Composite metric | Weighted combination of area, redness, granulation |

---

## 🔑 Key Technical Decisions

### Why Otsu Thresholding (Not Deep Learning)?
- **Hackathon timeframe**: 24-hour constraint
- **No labeled dataset**: Medical wound segmentation requires expert annotations
- **Interpretability**: Rule-based approach is explainable
- **Future work**: U-Net implementation planned with proper dataset

### Why Groq + Gemini?
- **Speed**: Groq's Llama 3.1 offers sub-second inference
- **Redundancy**: Gemini as fallback ensures reliability
- **Cost**: Both offer free tiers for prototyping

### Why Pathway?
- **Streaming-first**: Built for real-time data pipelines
- **Hackathon sponsor**: Technical support and integration help

---

## 📁 Project Structure

```
meditrack/
├── src/meditrack/
│   ├── cv/
│   │   ├── preprocessing.py    # Image loading and normalization
│   │   ├── segmentation.py     # Wound detection (thresholding)
│   │   └── postprocessing.py   # Mask cleanup
│   ├── llm/
│   │   ├── ai_client.py        # Groq & Gemini API integration
│   │   └── analyzer.py         # Metric-to-text conversion
│   └── pipeline/
│       └── pathway_pipeline.py # Real-time streaming
├── streamlit_app_enhanced.py   # Main dashboard
├── aparavi_integration.py      # PHI detection (demo mode)
├── requirements.txt
└── README.md
```

---

## 🎯 Roadmap

### ✅ Completed (Hackathon)
- [x] OpenCV-based wound detection
- [x] LLM integration (Groq/Gemini)
- [x] Pathway streaming pipeline
- [x] Streamlit dashboard
- [x] Basic metrics extraction

### 🔄 In Progress
- [ ] **U-Net deep learning model** - Training on wound segmentation dataset
- [ ] REST API with FastAPI
- [ ] Docker containerization

### 📋 Future Work
- [ ] Mobile app (React Native)
- [ ] Clinical validation study
- [ ] HIPAA compliance audit
- [ ] Multi-language support

---

## ⚠️ Important Disclaimer

```
┌───────────────────────────────────────────────────────────────────┐
│  🚨 EDUCATIONAL PROTOTYPE ONLY - NOT A MEDICAL DEVICE             │
│                                                                   │
│  MediTrack is a hackathon project for demonstration purposes.     │
│                                                                   │
│  • ❌ NOT FDA approved or cleared                                 │
│  • ❌ NOT for clinical diagnosis or treatment                     │
│  • ❌ NOT a replacement for professional medical advice           │
│                                                                   │
│  Always consult qualified healthcare professionals.               │
└───────────────────────────────────────────────────────────────────┘
```

---

## 👥 Team

**Hack With Chicago 2.0** | November 2024

- **Meenakshi Sridharan Sundaram** - [GitHub](https://github.com/Msundara19) | [LinkedIn](https://linkedin.com/in/meenakshi-sridharan)
- **Akshitha Priadharshini** - Team Member

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with ❤️ for Hack With Chicago 2.0**

*Empowering patients and providers with AI-driven wound care insights*

</div>
