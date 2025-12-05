"""
Pose Detector Module

Uses MediaPipe to detect body keypoints from video frames.
Optimized for golf swing analysis with focus on key body landmarks.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional


class PoseDetector:
    """
    Detects human pose using MediaPipe Pose.
    
    Attributes:
        mp_pose: MediaPipe Pose object
        pose: Pose detector instance
        mp_drawing: MediaPipe drawing utilities
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the pose detector.
        
        Args:
            static_image_mode: Whether to treat each image independently
            model_complexity: 0=lite, 1=full, 2=heavy (use 1 for balance)
            smooth_landmarks: Whether to smooth landmarks across frames
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Key landmarks for golf swing analysis
        self.golf_landmarks = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[object], np.ndarray]:
        """
        Process a single frame to detect pose landmarks.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (pose_landmarks, processed_frame)
            pose_landmarks: MediaPipe pose landmarks object (or None if not detected)
            processed_frame: RGB frame
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        return results.pose_landmarks, frame_rgb
    
    def extract_landmarks(self, pose_landmarks: object, frame_shape: Tuple[int, int]) -> Dict[str, Tuple[float, float, float]]:
        """
        Extract key landmarks as pixel coordinates.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks object
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Dictionary mapping landmark names to (x, y, visibility) tuples
        """
        if pose_landmarks is None:
            return {}
        
        height, width = frame_shape[:2]
        landmarks_dict = {}
        
        for name, idx in self.golf_landmarks.items():
            landmark = pose_landmarks.landmark[idx]
            # Convert normalized coordinates to pixels
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            visibility = landmark.visibility
            
            landmarks_dict[name] = (x, y, visibility)
        
        return landmarks_dict
    
    def draw_pose(
        self,
        frame: np.ndarray,
        pose_landmarks: object,
        draw_full_skeleton: bool = True
    ) -> np.ndarray:
        """
        Draw pose landmarks and connections on the frame.
        
        Args:
            frame: Input frame (RGB or BGR)
            pose_landmarks: MediaPipe pose landmarks object
            draw_full_skeleton: Whether to draw full skeleton or just key points
            
        Returns:
            Frame with drawn pose overlay
        """
        if pose_landmarks is None:
            return frame
        
        annotated_frame = frame.copy()
        
        if draw_full_skeleton:
            # Draw the pose annotation on the frame
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            # Draw only golf-relevant landmarks
            for name, idx in self.golf_landmarks.items():
                landmark = pose_landmarks.landmark[idx]
                height, width = frame.shape[:2]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Draw circle at landmark
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                # Draw landmark name
                cv2.putText(
                    annotated_frame,
                    name.replace('_', ' ').title(),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
        
        return annotated_frame
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 1
    ) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Process entire video and extract landmarks from all frames.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated video
            skip_frames: Process every Nth frame (1 = all frames)
            
        Returns:
            Tuple of (landmarks_list, annotated_frames)
            landmarks_list: List of landmark dictionaries for each frame
            annotated_frames: List of frames with pose overlay
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        landmarks_list = []
        annotated_frames = []
        
        # Setup video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames if specified
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # Process frame
            pose_landmarks, frame_rgb = self.process_frame(frame)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(pose_landmarks, frame.shape)
            landmarks_list.append(landmarks)
            
            # Draw pose
            annotated_frame = self.draw_pose(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), pose_landmarks)
            annotated_frames.append(annotated_frame)
            
            # Write to output video
            if output_path:
                out.write(annotated_frame)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        
        return landmarks_list, annotated_frames
    
    def get_confidence_score(self, pose_landmarks: object) -> float:
        """
        Calculate average visibility/confidence score for detected pose.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks object
            
        Returns:
            Average visibility score (0-1)
        """
        if pose_landmarks is None:
            return 0.0
        
        visibilities = [
            pose_landmarks.landmark[idx].visibility
            for idx in self.golf_landmarks.values()
        ]
        
        return np.mean(visibilities)
    
    def close(self):
        """Clean up resources."""
        self.pose.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def resize_frame(frame: np.ndarray, max_width: int = 720) -> np.ndarray:
    """
    Resize frame to reduce computational load while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width for resized frame
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return frame
