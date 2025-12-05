# ⛳ Golf Swing Analyzer - Quick Start Guide

## 🎉 Your App is Ready!

The Golf Swing Analyzer is now fully built and running at:
**http://localhost:8501**

## 📁 Project Structure

```
athena/
├── app.py                      # Main Streamlit application ✅
├── pyproject.toml              # Project dependencies (uv) ✅
├── requirements.txt            # Deployment dependencies ✅
├── README.md                   # Project documentation ✅
├── DEPLOYMENT.md              # Deployment guide ✅
├── .gitignore                 # Git ignore rules ✅
├── utils/                     # Core modules
│   ├── __init__.py           # Package initializer ✅
│   ├── pose_detector.py      # MediaPipe pose detection ✅
│   ├── angle_calculator.py   # Angle calculations ✅
│   ├── swing_analyzer.py     # Scoring logic ✅
│   └── visualizer.py         # Visualizations ✅
├── sample_videos/            # Sample golf videos
│   └── README.md            # Guide to get videos ✅
└── .venv/                   # Virtual environment (uv)
```

## 🚀 How to Use

### 1. Run the App (Already Running!)

The app is currently running at http://localhost:8501

To start it manually in the future:
```bash
cd /Users/alokmaurya/workspace/athena
source .venv/bin/activate
streamlit run app.py
```

### 2. Get a Golf Swing Video

**Option A: Record Yourself**
- Use your phone camera
- Side view, 10-15 feet away
- Full body visible
- 5-15 seconds

**Option B: Download from YouTube**
```bash
# Install yt-dlp (if not installed)
pip install yt-dlp

# Download a golf swing video
yt-dlp -f "best[height<=720]" -o "sample_videos/swing.mp4" "YOUTUBE_URL"
```

Search YouTube for: "golf swing slow motion side view"

**Option C: Use Stock Footage**
- Pexels.com - Search "golf swing"
- Pixabay.com - Free golf videos

### 3. Analyze Your Swing

1. Open http://localhost:8501 in your browser
2. Click "Browse files" or drag & drop a video
3. Click "🚀 Analyze Swing"
4. Wait for processing (usually 10-30 seconds)
5. Review your results!

## 📊 What You'll Get

- ✅ **Overall Score** (0-100)
- ✅ **Score Breakdown**: Posture, Rotation, Tempo, Balance
- ✅ **Visual Feedback**: Pose overlay on video
- ✅ **Angle Graphs**: Hip rotation, shoulder rotation, knee bend, etc.
- ✅ **Key Metrics**: Max rotations, X-factor, spine tilt
- ✅ **Actionable Feedback**: Specific improvement recommendations
- ✅ **Downloadable Report**: Save your analysis

## 🛠️ Development Commands

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Add a new package
uv add package-name

# Remove a package
uv remove package-name

# Update dependencies
uv sync --upgrade
```

### Run the App

```bash
# Development mode (auto-reload on changes)
streamlit run app.py

# Production mode
streamlit run app.py --server.headless true --server.port 8501
```

### Stop the App

Press `Ctrl+C` in the terminal where Streamlit is running.

## 🌐 Deploy to Production

Follow the comprehensive guide in **DEPLOYMENT.md**:

### Quick Deploy Steps:

1. **Create GitHub Repo**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/golf-swing-analyzer.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Click "Deploy"

3. **Get Your Live URL**
   - https://your-app-name.streamlit.app
   - Share with anyone!

## 🎯 Features Implemented

### Core Features ✅
- [x] Video upload (MP4, MOV, AVI)
- [x] MediaPipe pose detection
- [x] Angle calculation (11+ angles)
- [x] Biomechanics-based scoring
- [x] Visual pose overlay
- [x] Interactive angle graphs (Plotly)
- [x] Score breakdown radar chart
- [x] Metrics comparison
- [x] Actionable feedback
- [x] Downloadable report

### Technical Features ✅
- [x] Optimized for limited resources
- [x] Frame resizing for efficiency
- [x] Angle smoothing (moving average)
- [x] Progress indicators
- [x] Error handling
- [x] Responsive UI
- [x] Clean code structure
- [x] Type hints
- [x] Documentation

## 🎓 Understanding the Scores

### Posture Score (0-100)
- Spine angle at address
- Knee bend consistency
- Upper body alignment

**Ideal Ranges:**
- Spine tilt: 25-45° from vertical
- Knee bend: 140-170° (slight flex)

### Rotation Score (0-100)
- Hip rotation range
- Shoulder rotation range
- X-factor (shoulder-hip separation)

**Ideal Ranges:**
- Hip rotation: 45-90°
- Shoulder rotation: 90-110°
- X-factor: 15-45°

### Tempo Score (0-100)
- Swing smoothness
- Consistent speed
- No jerky movements

**Ideal:**
- Smooth acceleration
- 1-1.5 second total swing
- Low velocity variance

### Balance Score (0-100)
- Head stability
- Weight distribution
- Stable base

**Ideal:**
- Minimal head movement
- Shoulder-width stance
- Controlled finish

## 🐛 Troubleshooting

### App won't start
```bash
# Reinstall dependencies
uv sync

# Try with fresh environment
rm -rf .venv
uv sync
```

### "Module not found" error
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Check installed packages
uv pip list
```

### Video processing fails
- Check video format (MP4, MOV, AVI)
- Try resizing video to 720p
- Ensure video is < 50MB
- Check video has clear view of person

### Pose detection not working
- Ensure good lighting in video
- Full body must be visible
- Camera should be 10-15 feet away
- Try side-view angle

## 💡 Tips for Best Results

### Video Quality
- Clear, unobstructed view
- Good lighting (natural light ideal)
- Stable camera (use tripod)
- Side-on or face-on angle
- Full body in frame

### Swing Recording
- Waist-height camera
- 10-15 feet distance
- Capture address to follow-through
- 5-15 second duration
- Avoid background clutter

### Performance
- Videos auto-resize to 720p
- Frame skipping for speed
- Recommended: 30 FPS videos
- Limit video to 15 seconds

## 📚 Resources

- **MediaPipe**: https://google.github.io/mediapipe/
- **Streamlit**: https://docs.streamlit.io/
- **Golf Biomechanics**: Study professional swing mechanics
- **Sample Videos**: YouTube "golf swing analysis"

## 🎬 Next Steps

1. **Test the app** with different videos
2. **Get sample videos** (see sample_videos/README.md)
3. **Deploy to cloud** (see DEPLOYMENT.md)
4. **Share with friends** and get feedback
5. **Add to portfolio/resume**

## 🤝 Enhancements (Future Ideas)

Want to improve the app? Here are ideas:

- [ ] Camera recording via browser
- [ ] Multiple swing comparison
- [ ] AI-based recommendations
- [ ] Slow-motion playback
- [ ] Export annotated video
- [ ] Swing library/history
- [ ] Professional swing comparison
- [ ] Club-specific analysis
- [ ] 3D visualization
- [ ] Mobile app version

## 📝 Notes

- **Built with**: Python 3.12, uv package manager
- **Optimized for**: Limited computational resources
- **Perfect for**: Portfolio, learning, demos
- **Not for**: Professional golf instruction (consult a pro!)

## 🙌 You're All Set!

Your Golf Swing Analyzer is ready to use! Open http://localhost:8501 and start analyzing swings!

For deployment to production, see **DEPLOYMENT.md**.

**Happy Analyzing!** ⛳🏌️‍♂️
