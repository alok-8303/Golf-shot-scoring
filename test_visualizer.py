"""
Test script for visualizer enhancements
Tests the new visual feedback features without needing a full video
"""

import sys
import os

# Prevent Streamlit from running
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import numpy as np
import cv2
from utils.visualizer import SwingVisualizer

# Create dummy test data
def test_visualizations():
    print("Testing enhanced visualizations...")
    
    # Create dummy frames (simple colored rectangles)
    height, width = 480, 640
    start_frame = np.zeros((height, width, 3), dtype=np.uint8)
    start_frame[:, :] = (50, 100, 50)  # Green tint
    
    end_frame = np.zeros((height, width, 3), dtype=np.uint8)
    end_frame[:, :] = (50, 50, 100)  # Blue tint
    
    # Add text to frames
    cv2.putText(start_frame, "START POSITION", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(end_frame, "END POSITION", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create dummy landmarks
    start_landmarks = {
        'left_shoulder': (200, 200, 0.9),
        'right_shoulder': (400, 200, 0.9),
        'left_hip': (220, 350, 0.9),
        'right_hip': (380, 350, 0.9),
    }
    
    end_landmarks = {
        'left_shoulder': (250, 200, 0.9),
        'right_shoulder': (450, 200, 0.9),
        'left_hip': (240, 350, 0.9),
        'right_hip': (400, 350, 0.9),
    }
    
    # Create dummy angles
    start_angles = {
        'hip_rotation': 10.0,
        'shoulder_rotation': 15.0,
        'spine_tilt': 30.0,
        'left_knee': 160.0,
        'right_knee': 155.0,
    }
    
    end_angles = {
        'hip_rotation': 75.0,
        'shoulder_rotation': 95.0,
        'spine_tilt': 35.0,
        'left_knee': 140.0,
        'right_knee': 165.0,
    }
    
    # Test rotation indicators
    print("✓ Testing rotation indicators...")
    start_with_rotation = SwingVisualizer.draw_rotation_indicators(
        start_frame.copy(), start_landmarks, start_angles
    )
    end_with_rotation = SwingVisualizer.draw_rotation_indicators(
        end_frame.copy(), end_landmarks, end_angles
    )
    
    # Test swing comparison
    print("✓ Testing swing comparison...")
    comparison = SwingVisualizer.create_swing_comparison(
        start_with_rotation,
        end_with_rotation,
        start_angles,
        end_angles,
        start_landmarks,
        end_landmarks
    )
    
    # Save test output
    output_path = "test_comparison.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"✅ Test complete! Output saved to {output_path}")
    print(f"   Image size: {comparison.shape}")
    
    return True

if __name__ == "__main__":
    test_visualizations()
