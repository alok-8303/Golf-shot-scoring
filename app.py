"""
Golf Swing Analyzer - Main Streamlit Application

A web application for analyzing golf swing videos using AI-powered pose detection.
Upload a golf swing video, get instant analysis with scores and feedback.
"""

import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from pathlib import Path

# Import our custom modules
from utils.pose_detector import PoseDetector, resize_frame
from utils.angle_calculator import AngleCalculator, SwingPhaseDetector
from utils.swing_analyzer import SwingAnalyzer
from utils.visualizer import SwingVisualizer


# Page configuration
st.set_page_config(
    page_title="⛳ Golf Swing Analyzer",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    .excellent {
        background-color: #d4edda;
        color: #155724;
    }
    .good {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .fair {
        background-color: #fff3cd;
        color: #856404;
    }
    .poor {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)


def process_video(video_file, progress_bar, status_text):
    """
    Process uploaded video and return analysis results.
    
    Args:
        video_file: Uploaded video file object
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text element
        
    Returns:
        Tuple of (swing_score, angles_over_time, annotated_frames, fps)
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name
    
    try:
        # Initialize components
        status_text.text("🔍 Initializing pose detector...")
        progress_bar.progress(10)
        
        pose_detector = PoseDetector(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        angle_calculator = AngleCalculator()
        swing_analyzer = SwingAnalyzer()
        
        # Process video
        status_text.text("📹 Processing video frames...")
        progress_bar.progress(30)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        landmarks_list = []
        angles_list = []
        annotated_frames = []
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for efficiency
            frame = resize_frame(frame, max_width=720)
            
            # Process frame
            pose_landmarks, frame_rgb = pose_detector.process_frame(frame)
            
            # Extract landmarks
            landmarks = pose_detector.extract_landmarks(pose_landmarks, frame.shape)
            landmarks_list.append(landmarks)
            
            # Calculate angles
            angles = angle_calculator.golf_swing_angles(landmarks)
            angles_list.append(angles)
            
            # Draw pose overlay
            annotated_frame = pose_detector.draw_pose(
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                pose_landmarks
            )
            
            # Add angle overlay
            if angles and landmarks:
                annotated_frame = SwingVisualizer.create_angle_overlay(
                    annotated_frame, angles, landmarks
                )
            
            annotated_frames.append(annotated_frame)
            
            # Update progress
            frame_count += 1
            progress = 30 + int((frame_count / total_frames) * 50)
            progress_bar.progress(min(progress, 80))
            
        cap.release()
        
        # Smooth angles
        status_text.text("📊 Analyzing swing mechanics...")
        progress_bar.progress(85)
        
        smoothed_angles = angle_calculator.smooth_angles(angles_list, window_size=5)
        
        # Analyze swing
        swing_score = swing_analyzer.analyze_swing(
            smoothed_angles,
            landmarks_list,
            fps=fps
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        pose_detector.close()
        
        return swing_score, smoothed_angles, annotated_frames, fps, landmarks_list
        
    finally:
        # Cleanup temporary file
        if os.path.exists(video_path):
            os.remove(video_path)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">⛳ Golf Swing Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered golf swing analysis using pose detection</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info(
            """
            This app uses **MediaPipe** pose detection to analyze golf swings.
            
            **Features:**
            - Pose detection and tracking
            - Angle measurements
            - Biomechanics-based scoring
            - Visual feedback
            
            **How to use:**
            1. Upload a golf swing video
            2. Wait for processing
            3. Review your analysis!
            """
        )
        
        st.header("⚙️ Settings")
        show_original = st.checkbox("Show original video", value=True)
        show_skeleton = st.checkbox("Show full skeleton", value=True)
        
        st.header("📊 Scoring Criteria")
        st.markdown("""
        - **Posture** (25%): Spine angle, knee bend
        - **Rotation** (25%): Hip & shoulder turn
        - **Tempo** (25%): Swing smoothness
        - **Balance** (25%): Stability & head position
        """)
    
    # Main content area
    st.header("📹 Upload Your Golf Swing Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, MOV, AVI)",
        type=['mp4', 'mov', 'avi'],
        help="Upload a golf swing video for analysis. Best results with clear, side-view shots."
    )
    
    if uploaded_file is not None:
        # Create columns for video display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Uploaded Video")
            st.video(uploaded_file)
        
        # Process button
        if st.button("🚀 Analyze Swing", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process the video
                swing_score, angles_over_time, annotated_frames, fps, landmarks_list = process_video(
                    uploaded_file,
                    progress_bar,
                    status_text
                )
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Overall score with color coding
                st.header("🎯 Overall Score")
                
                score_class = "excellent" if swing_score.overall_score >= 90 else \
                              "good" if swing_score.overall_score >= 75 else \
                              "fair" if swing_score.overall_score >= 60 else "poor"
                
                st.markdown(
                    f'<div class="score-box {score_class}">{swing_score.overall_score}/100</div>',
                    unsafe_allow_html=True
                )
                
                # Score breakdown
                st.header("📊 Score Breakdown")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Posture", f"{swing_score.posture_score}/100")
                with col2:
                    st.metric("Rotation", f"{swing_score.rotation_score}/100")
                with col3:
                    st.metric("Tempo", f"{swing_score.tempo_score}/100")
                with col4:
                    st.metric("Balance", f"{swing_score.balance_score}/100")
                
                # Radar chart
                st.plotly_chart(
                    SwingVisualizer.plot_score_breakdown(
                        swing_score.posture_score,
                        swing_score.rotation_score,
                        swing_score.tempo_score,
                        swing_score.balance_score
                    ),
                    use_container_width=True
                )
                
                # Feedback
                st.header("💡 Feedback & Recommendations")
                
                for feedback_item in swing_score.feedback:
                    st.write(f"- {feedback_item}")
                
                # VISUAL FEEDBACK SECTION - Start vs End Swing Comparison
                st.header("👁️ Visual Feedback: Swing Comparison")
                st.markdown("""
                Compare your starting position (address) with your swing position (same frame as shown in video analysis). 
                Key rotation angles and spine tilt are highlighted below.
                """)
                
                if len(annotated_frames) >= 10 and len(angles_over_time) >= 10:
                    # Get start frame (address - early in swing)
                    start_idx = min(5, len(annotated_frames) - 1)
                    
                    # Use the SAME frame as the slider default (middle frame)
                    # This is exactly what's shown in "Annotated Video Analysis" section
                    end_idx = len(annotated_frames) // 2
                    
                    # Add rotation indicators to frames
                    start_frame_with_rotation = SwingVisualizer.draw_rotation_indicators(
                        annotated_frames[start_idx].copy(),
                        landmarks_list[start_idx],
                        angles_over_time[start_idx]
                    )
                    
                    end_frame_with_rotation = SwingVisualizer.draw_rotation_indicators(
                        annotated_frames[end_idx].copy(),
                        landmarks_list[end_idx],
                        angles_over_time[end_idx]
                    )
                    
                    # Create comparison image
                    comparison_img = SwingVisualizer.create_swing_comparison(
                        start_frame_with_rotation,
                        end_frame_with_rotation,
                        angles_over_time[start_idx],
                        angles_over_time[end_idx],
                        landmarks_list[start_idx],
                        landmarks_list[end_idx]
                    )
                    
                    # Display the comparison
                    st.image(
                        cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB),
                        caption=f"Address Position (Frame {start_idx+1}) vs Swing End at Peak Rotation (Frame {end_idx+1})",
                        use_container_width=True
                    )
                    
                    # Add interpretation
                    st.info("""
                    **How to Read This:**
                    - **START**: Your address position (setup)
                    - **END**: Mid-swing position (same as video slider default)
                    - **Yellow lines/arcs**: Hip rotation indicators
                    - **Orange lines/arcs**: Shoulder rotation indicators  
                    - **Purple line**: Spine angle
                    - **Metrics Table Headers**: START (green) and END (blue) with CHANGE values
                    """)
                    
                    # Show which frames are being compared
                    st.success(f"📊 Comparing frame {start_idx+1} (start) with frame {end_idx+1} (mid-swing)")
                else:
                    st.warning("⚠️ Not enough frames for visual comparison. Video may be too short.")
                
                # Key metrics
                if swing_score.metrics:
                    st.header("📏 Key Metrics")
                    
                    cols = st.columns(3)
                    metric_items = list(swing_score.metrics.items())
                    
                    for i, (metric_name, value) in enumerate(metric_items):
                        with cols[i % 3]:
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{value}°"
                            )
                    
                    # Metrics comparison chart
                    st.plotly_chart(
                        SwingVisualizer.plot_metrics_comparison(swing_score.metrics),
                        use_container_width=True
                    )
                
                # Angle graphs
                st.header("📈 Angle Analysis Over Time")
                
                st.plotly_chart(
                    SwingVisualizer.plot_angles_over_time(
                        angles_over_time,
                        fps=fps
                    ),
                    use_container_width=True
                )
                
                # Video comparison
                st.header("🎬 Annotated Video Analysis")
                
                st.info(f"📊 Total frames analyzed: {len(annotated_frames)} | FPS: {fps:.1f}")
                
                # Frame selector
                frame_index = st.slider(
                    "Select frame to view",
                    0,
                    len(annotated_frames) - 1,
                    len(annotated_frames) // 2
                )
                
                # Display selected frame with rotation indicators
                if frame_index < len(annotated_frames):
                    # Add rotation indicators to selected frame
                    frame_with_indicators = SwingVisualizer.draw_rotation_indicators(
                        annotated_frames[frame_index].copy(),
                        landmarks_list[frame_index],
                        angles_over_time[frame_index]
                    )
                    
                    st.image(
                        cv2.cvtColor(frame_with_indicators, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_index + 1}/{len(annotated_frames)} - with rotation & tilt indicators",
                        use_container_width=True
                    )
                    
                    # Show angles for this specific frame
                    if angles_over_time[frame_index]:
                        st.markdown("**Angles at this frame:**")
                        cols = st.columns(5)
                        angle_items = list(angles_over_time[frame_index].items())[:5]
                        for i, (angle_name, angle_val) in enumerate(angle_items):
                            with cols[i]:
                                st.metric(
                                    angle_name.replace('_', ' ').title(),
                                    f"{angle_val:.1f}°"
                                )
                
                # Download results
                st.header("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create summary text
                    summary = f"""
Golf Swing Analysis Report
==========================

Overall Score: {swing_score.overall_score}/100

Score Breakdown:
- Posture: {swing_score.posture_score}/100
- Rotation: {swing_score.rotation_score}/100
- Tempo: {swing_score.tempo_score}/100
- Balance: {swing_score.balance_score}/100

Key Metrics:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v}°" for k, v in swing_score.metrics.items()])}

Feedback:
{chr(10).join([f"- {fb}" for fb in swing_score.feedback])}

Generated by Golf Swing Analyzer
"""
                    
                    st.download_button(
                        label="📄 Download Report (TXT)",
                        data=summary,
                        file_name="golf_swing_report.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.exception(e)
    
    else:
        # Instructions when no video is uploaded
        st.info("👆 Upload a golf swing video to get started!")
        
        st.header("📖 Tips for Best Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Video Requirements:**
            - Clear, unobstructed view
            - Side-on angle (face-on also works)
            - Good lighting
            - Full body visible
            - 5-15 seconds duration
            """)
        
        with col2:
            st.markdown("""
            **What We Analyze:**
            - Body posture & alignment
            - Hip & shoulder rotation
            - Knee flex & balance
            - Swing tempo & rhythm
            - Overall biomechanics
            """)
        
        st.header("🎥 Sample Videos")
        st.markdown("""
        Don't have a video? You can:
        - Record yourself using a phone camera
        - Use a friend to record your swing
        - Download sample golf swings from YouTube
        
        For best results, place the camera at waist height, about 10-15 feet away.
        """)


if __name__ == "__main__":
    main()
