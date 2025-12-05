"""
Visualizer Module

Creates visualizations for golf swing analysis including:
- Pose overlays on video frames
- Angle graphs over time
- Comparison charts
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import io
from PIL import Image


class SwingVisualizer:
    """
    Creates visualizations for golf swing analysis.
    """
    
    @staticmethod
    def create_side_by_side_comparison(
        original_frame: np.ndarray,
        annotated_frame: np.ndarray,
        title: str = "Comparison"
    ) -> np.ndarray:
        """
        Create side-by-side comparison of original and annotated frames.
        
        Args:
            original_frame: Original video frame
            annotated_frame: Frame with pose overlay
            title: Title for the comparison
            
        Returns:
            Combined frame with both views
        """
        height, width = original_frame.shape[:2]
        
        # Create canvas for side-by-side
        combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Place frames
        combined[:, :width] = original_frame
        combined[:, width:] = annotated_frame
        
        # Add labels
        cv2.putText(combined, "Original", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Pose Analysis", (width + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    @staticmethod
    def create_swing_comparison(
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        start_angles: Dict[str, float],
        end_angles: Dict[str, float],
        start_landmarks: Dict,
        end_landmarks: Dict
    ) -> np.ndarray:
        """
        Create visual comparison of start swing (address) vs mid-swing position.
        Shows both poses with key metrics overlaid.
        
        Args:
            start_frame: Frame from address position (annotated)
            end_frame: Frame from mid-swing position (annotated)
            start_angles: Angles at address
            end_angles: Angles at mid-swing
            start_landmarks: Landmarks at address
            end_landmarks: Landmarks at mid-swing
            
        Returns:
            Combined comparison image with metrics
        """
        height, width = start_frame.shape[:2]
        
        # Create canvas for side-by-side (with extra space at bottom for metrics)
        combined = np.ones((height + 200, width * 2, 3), dtype=np.uint8) * 255
        
        # Place frames
        combined[0:height, 0:width] = start_frame
        combined[0:height, width:width*2] = end_frame
        
        # Add titles with background - extra tall to ensure no overlap
        title_height = 70
        cv2.rectangle(combined, (0, 0), (width, title_height), (0, 100, 0), -1)
        cv2.rectangle(combined, (width, 0), (width*2, title_height), (0, 0, 150), -1)
        
        cv2.putText(combined, "START: Address Position", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "END: Mid-Swing Position", (width + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw rotation arrows on the frames
        if start_landmarks and end_landmarks:
            # Draw hip rotation indicator on start frame
            if 'left_hip' in start_landmarks and 'right_hip' in start_landmarks:
                left_hip = start_landmarks['left_hip'][:2]
                right_hip = start_landmarks['right_hip'][:2]
                center_x = int((left_hip[0] + right_hip[0]) / 2)
                center_y = int((left_hip[1] + right_hip[1]) / 2)
                
                # Draw rotation arc
                cv2.ellipse(combined, (center_x, center_y), (50, 30), 0, 0, 180, (0, 255, 255), 3)
                cv2.putText(combined, "Hip", (center_x - 20, center_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw hip rotation indicator on end frame
            if 'left_hip' in end_landmarks and 'right_hip' in end_landmarks:
                left_hip = end_landmarks['left_hip'][:2]
                right_hip = end_landmarks['right_hip'][:2]
                center_x = int((left_hip[0] + right_hip[0]) / 2) + width
                center_y = int((left_hip[1] + right_hip[1]) / 2)
                
                # Draw rotation arc
                cv2.ellipse(combined, (center_x, center_y), (50, 30), 0, 0, 180, (0, 255, 255), 3)
                cv2.putText(combined, "Hip", (center_x - 20, center_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add metrics comparison at the bottom
        metrics_y_start = height + 10
        
        # Background for metrics
        cv2.rectangle(combined, (0, height), (width*2, height + 200), (240, 240, 240), -1)
        
        # Title for metrics
        cv2.putText(combined, "KEY METRICS COMPARISON", (width//2 + 50, metrics_y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Column headers - wider spacing for ideal range column
        header_y = metrics_y_start + 50
        col_width = width * 2 // 3
        
        # Draw column headers
        cv2.putText(combined, "START", (col_width - 80, header_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
        cv2.putText(combined, "END", (col_width + 80, header_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 0), 2)
        cv2.putText(combined, "CHANGE", (col_width + 220, header_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
        cv2.putText(combined, "IDEAL RANGE", (col_width + 360, header_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
        
        # Draw underline for headers
        cv2.line(combined, (20, header_y + 5), (width*2 - 20, header_y + 5), (150, 150, 150), 1)
        
        # Define metrics to display with ideal ranges
        # Ideal ranges based on golf biomechanics best practices
        metrics_to_show = [
            ('hip_rotation', 'Hip Rotation', 'deg', '40-50 deg'),
            ('shoulder_rotation', 'Shoulder Rotation', 'deg', '80-100 deg'),
            ('spine_tilt', 'Spine Tilt', 'deg', '25-35 deg'),
            ('left_knee', 'Left Knee Angle', 'deg', 'Stable +/-5'),
            ('right_knee', 'Right Knee Angle', 'deg', 'Stable +/-5'),
        ]
        
        y_offset = header_y + 25
        
        for i, (key, label, unit, ideal_range) in enumerate(metrics_to_show):
            if i >= 5:  # Limit to 5 metrics
                break
            
            start_val = start_angles.get(key, 0)
            end_val = end_angles.get(key, 0)
            delta = end_val - start_val
            
            # Draw metric row
            row_y = y_offset + (i * 25)
            
            # Metric name
            cv2.putText(combined, f"{label}:", (30, row_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Start value (green)
            cv2.putText(combined, f"{start_val:.1f}{unit}", (col_width - 80, row_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
            
            # End value (blue)
            cv2.putText(combined, f"{end_val:.1f}{unit}", (col_width + 80, row_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 2)
            
            # Change (delta)
            delta_color = (0, 180, 0) if abs(delta) > 10 else (100, 100, 100)
            delta_text = f"{delta:+.1f}{unit}"
            cv2.putText(combined, delta_text, (col_width + 220, row_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, delta_color, 1)
            
            # Ideal range (blue)
            cv2.putText(combined, ideal_range, (col_width + 360, row_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        
        return combined
    
    @staticmethod
    def draw_rotation_indicators(
        frame: np.ndarray,
        landmarks: Dict[str, Tuple[float, float, float]],
        angles: Dict[str, float]
    ) -> np.ndarray:
        """
        Draw visual indicators for rotation angles on the frame.
        
        Args:
            frame: Input frame
            landmarks: Dictionary of landmark positions
            angles: Dictionary of calculated angles
            
        Returns:
            Frame with rotation indicators
        """
        annotated = frame.copy()
        
        # Draw hip rotation indicator
        if 'left_hip' in landmarks and 'right_hip' in landmarks and 'hip_rotation' in angles:
            left_hip = landmarks['left_hip'][:2]
            right_hip = landmarks['right_hip'][:2]
            
            center_x = int((left_hip[0] + right_hip[0]) / 2)
            center_y = int((left_hip[1] + right_hip[1]) / 2)
            
            # Draw line connecting hips
            cv2.line(annotated, tuple(map(int, left_hip)), tuple(map(int, right_hip)), 
                    (0, 255, 255), 3)
            
            # Draw rotation arc
            rotation = angles['hip_rotation']
            cv2.ellipse(annotated, (center_x, center_y), (60, 40), 0, 
                       -abs(rotation), abs(rotation), (0, 255, 255), 2)
            
            # Add text
            cv2.putText(annotated, f"Hip: {rotation:.1f}deg", 
                       (center_x - 60, center_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw shoulder rotation indicator
        if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks and 'shoulder_rotation' in angles:
            left_shoulder = landmarks['left_shoulder'][:2]
            right_shoulder = landmarks['right_shoulder'][:2]
            
            center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
            center_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
            
            # Draw line connecting shoulders
            cv2.line(annotated, tuple(map(int, left_shoulder)), tuple(map(int, right_shoulder)),
                    (255, 165, 0), 3)
            
            # Draw rotation arc
            rotation = angles['shoulder_rotation']
            cv2.ellipse(annotated, (center_x, center_y), (60, 40), 0,
                       -abs(rotation), abs(rotation), (255, 165, 0), 2)
            
            # Add text
            cv2.putText(annotated, f"Shoulder: {rotation:.1f}deg",
                       (center_x - 80, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Draw spine tilt indicator
        if 'spine_tilt' in angles and 'left_shoulder' in landmarks and 'left_hip' in landmarks:
            left_shoulder = landmarks['left_shoulder'][:2]
            left_hip = landmarks['left_hip'][:2]
            
            # Calculate spine midpoint
            spine_mid_x = int((left_shoulder[0] + left_hip[0]) / 2)
            spine_mid_y = int((left_shoulder[1] + left_hip[1]) / 2)
            
            # Draw spine line
            cv2.line(annotated, tuple(map(int, left_shoulder)), tuple(map(int, left_hip)),
                    (255, 0, 255), 3)
            
            # Draw vertical reference
            cv2.line(annotated, (spine_mid_x, spine_mid_y - 50), 
                    (spine_mid_x, spine_mid_y + 50), (200, 200, 200), 2, cv2.LINE_AA)
            
            # Add text
            tilt = angles['spine_tilt']
            cv2.putText(annotated, f"Spine: {tilt:.1f}deg",
                       (spine_mid_x + 10, spine_mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return annotated
    
    @staticmethod
    def create_angle_overlay(
        frame: np.ndarray,
        angles: Dict[str, float],
        landmarks: Dict[str, Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Draw angle measurements on the frame.
        
        Args:
            frame: Input frame
            angles: Dictionary of calculated angles
            landmarks: Dictionary of landmark positions
            
        Returns:
            Frame with angle annotations
        """
        annotated = frame.copy()
        
        # Define positions for angle text
        y_offset = 30
        
        # Key angles to display
        display_angles = [
            ('hip_rotation', 'Hip Rotation'),
            ('shoulder_rotation', 'Shoulder Rotation'),
            ('left_knee', 'Left Knee'),
            ('right_knee', 'Right Knee'),
            ('spine_tilt', 'Spine Tilt')
        ]
        
        for i, (angle_key, label) in enumerate(display_angles):
            if angle_key in angles:
                text = f"{label}: {angles[angle_key]:.1f}deg"
                y_pos = y_offset + (i * 30)
                
                # Background rectangle for text
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated,
                    (10, y_pos - text_height - 5),
                    (text_width + 20, y_pos + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated,
                    text,
                    (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
        
        return annotated
    
    @staticmethod
    def plot_angles_over_time(
        angles_over_time: List[Dict[str, float]],
        fps: float = 30.0,
        selected_angles: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create interactive plot of angles over time using Plotly.
        
        Args:
            angles_over_time: List of angle dictionaries
            fps: Frames per second
            selected_angles: List of angle names to plot (None = all)
            
        Returns:
            Plotly figure object
        """
        if not angles_over_time:
            return go.Figure()
        
        # Default angles to plot
        if selected_angles is None:
            selected_angles = [
                'hip_rotation',
                'shoulder_rotation',
                'left_knee',
                'right_knee',
                'spine_tilt'
            ]
        
        # Create time axis
        time_points = [i / fps for i in range(len(angles_over_time))]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each angle
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, angle_name in enumerate(selected_angles):
            angle_values = [
                frame.get(angle_name, None)
                for frame in angles_over_time
            ]
            
            # Filter out None values
            valid_indices = [j for j, v in enumerate(angle_values) if v is not None]
            valid_times = [time_points[j] for j in valid_indices]
            valid_angles = [angle_values[j] for j in valid_indices]
            
            if valid_angles:
                fig.add_trace(go.Scatter(
                    x=valid_times,
                    y=valid_angles,
                    mode='lines+markers',
                    name=angle_name.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
        
        # Update layout
        fig.update_layout(
            title='Angle Measurements Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Angle (degrees)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def plot_score_breakdown(
        posture_score: float,
        rotation_score: float,
        tempo_score: float,
        balance_score: float
    ) -> go.Figure:
        """
        Create radar chart showing score breakdown.
        
        Args:
            posture_score: Posture score (0-100)
            rotation_score: Rotation score (0-100)
            tempo_score: Tempo score (0-100)
            balance_score: Balance score (0-100)
            
        Returns:
            Plotly figure object
        """
        categories = ['Posture', 'Rotation', 'Tempo', 'Balance']
        scores = [posture_score, rotation_score, tempo_score, balance_score]
        
        fig = go.Figure()
        
        # Add actual scores
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='Your Score',
            line=dict(color='#1f77b4', width=2),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        # Add perfect score reference
        perfect = [100] * 5
        fig.add_trace(go.Scatterpolar(
            r=perfect,
            theta=categories + [categories[0]],
            fill='toself',
            name='Perfect Score',
            line=dict(color='#2ca02c', width=1, dash='dash'),
            fillcolor='rgba(44, 160, 44, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title='Swing Score Breakdown',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(metrics: Dict[str, float]) -> go.Figure:
        """
        Create bar chart comparing metrics to ideal ranges.
        
        Args:
            metrics: Dictionary of measured metrics
            
        Returns:
            Plotly figure object
        """
        metric_names = []
        actual_values = []
        ideal_mins = []
        ideal_maxs = []
        
        # Define ideal ranges
        ideal_ranges = {
            'max_hip_rotation': (45, 90),
            'max_shoulder_rotation': (90, 110),
            'x_factor': (15, 45),
            'address_spine_tilt': (25, 45),
        }
        
        for key, value in metrics.items():
            if key in ideal_ranges:
                metric_names.append(key.replace('_', ' ').title())
                actual_values.append(value)
                ideal_mins.append(ideal_ranges[key][0])
                ideal_maxs.append(ideal_ranges[key][1])
        
        if not metric_names:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Bar(
            name='Your Measurement',
            x=metric_names,
            y=actual_values,
            marker_color='#1f77b4'
        ))
        
        # Add ideal range (min)
        fig.add_trace(go.Scatter(
            name='Ideal Range Min',
            x=metric_names,
            y=ideal_mins,
            mode='markers',
            marker=dict(color='#2ca02c', size=10, symbol='diamond'),
        ))
        
        # Add ideal range (max)
        fig.add_trace(go.Scatter(
            name='Ideal Range Max',
            x=metric_names,
            y=ideal_maxs,
            mode='markers',
            marker=dict(color='#d62728', size=10, symbol='diamond'),
        ))
        
        fig.update_layout(
            title='Key Metrics vs. Ideal Ranges',
            xaxis_title='Metric',
            yaxis_title='Value (degrees)',
            barmode='group',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_progress_indicator(
        frame_number: int,
        total_frames: int,
        phase: str = ""
    ) -> np.ndarray:
        """
        Create a simple progress indicator overlay.
        
        Args:
            frame_number: Current frame number
            total_frames: Total number of frames
            phase: Current swing phase name
            
        Returns:
            Small progress bar image
        """
        width, height = 400, 60
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw background
        cv2.rectangle(img, (10, 10), (width - 10, 40), (50, 50, 50), -1)
        
        # Draw progress bar
        progress = int((frame_number / total_frames) * (width - 20))
        cv2.rectangle(img, (10, 10), (10 + progress, 40), (0, 255, 0), -1)
        
        # Add text
        text = f"Frame {frame_number}/{total_frames}"
        if phase:
            text += f" - {phase}"
        
        cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        
        return img
    
    @staticmethod
    def create_summary_image(
        overall_score: float,
        feedback: List[str],
        metrics: Dict[str, float]
    ) -> np.ndarray:
        """
        Create a summary image with scores and feedback.
        
        Args:
            overall_score: Overall swing score
            feedback: List of feedback strings
            metrics: Dictionary of metrics
            
        Returns:
            Summary image as numpy array
        """
        width, height = 800, 600
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw title
        cv2.putText(img, "Golf Swing Analysis Summary",
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 0, 0), 2)
        
        # Draw overall score
        score_color = (0, 255, 0) if overall_score >= 75 else (0, 165, 255) if overall_score >= 60 else (0, 0, 255)
        cv2.putText(img, f"Overall Score: {overall_score:.1f}/100",
                   (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, score_color, 3)
        
        # Draw feedback
        y_pos = 180
        cv2.putText(img, "Feedback:", (50, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        y_pos += 40
        
        for fb in feedback[:8]:  # Limit to 8 items
            cv2.putText(img, f"- {fb}", (70, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
            y_pos += 30
        
        return img
    
    @staticmethod
    def frames_to_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0
    ):
        """
        Save a list of frames as a video file.
        
        Args:
            frames: List of frame arrays
            output_path: Path to save video
            fps: Frames per second
        """
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
