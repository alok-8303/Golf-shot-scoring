# ⛳ Golf Swing Analyzer

A Python + Streamlit web application that analyzes golf swing videos using AI-powered pose detection and provides real-time feedback on swing mechanics.

## 🎯 Features

- **Video Upload**: Upload golf swing videos (MP4, MOV, AVI)
- **Pose Detection**: Uses MediaPipe to detect body keypoints
- **Swing Analysis**: Calculates key angles and metrics
  - Shoulder rotation
  - Hip rotation
  - Knee bend
  - Arm extension
  - Follow-through
- **Visual Feedback**: Overlayed skeleton on video frames
- **Scoring System**: 0-100 score with detailed feedback
- **Interactive Charts**: Angle measurements over time

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd athena
```

2. **Install dependencies using uv**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

3. **Run the application**
```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Run Streamlit app
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
athena/
├── app.py                    # Main Streamlit application
├── pyproject.toml            # Project dependencies (uv)
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── utils/
│   ├── __init__.py          # Package initializer
│   ├── pose_detector.py     # MediaPipe pose detection
│   ├── angle_calculator.py  # Angle computation logic
│   ├── swing_analyzer.py    # Swing scoring & analysis
│   └── visualizer.py        # Video overlay & charts
├── sample_videos/           # Sample golf swing videos
└── assets/                  # Images, logos, etc.
```

## 🛠️ Technology Stack

- **Streamlit**: Web framework for Python
- **MediaPipe**: Google's pose detection library
- **OpenCV**: Video processing
- **NumPy**: Numerical computations
- **Matplotlib/Plotly**: Data visualization

## 📊 How It Works

1. **Upload**: User uploads a golf swing video
2. **Detection**: MediaPipe detects body keypoints in each frame
3. **Analysis**: System calculates angles and identifies swing phases
4. **Scoring**: Biomechanics-based algorithm scores the swing
5. **Feedback**: Visual overlay + detailed report with recommendations

## 🎓 Scoring Criteria

- **Posture Score** (25%): Spine angle, knee bend, balance
- **Rotation Score** (25%): Hip and shoulder turn
- **Tempo Score** (25%): Swing smoothness and consistency
- **Follow-through Score** (25%): Arm extension and finish position

## 🌐 Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click!

### Generate requirements.txt for deployment
```bash
uv pip compile pyproject.toml -o requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - feel free to use this project for learning and portfolio purposes.

## 👨‍💻 Author

@Alok Maurya

## 🙏 Acknowledgments

- MediaPipe by Google
- Streamlit team
- Golf biomechanics research community

---

**Note**: This is a prototype designed to run on limited computational resources. For production use, consider GPU acceleration and professional biomechanics consultation.
