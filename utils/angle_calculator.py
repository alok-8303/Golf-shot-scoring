"""
Angle Calculator Module

Calculates angles between body joints for golf swing analysis.
Uses vector geometry to compute angles from pose landmarks.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import math


class AngleCalculator:
    """
    Calculates angles between body joints for biomechanical analysis.
    """
    
    @staticmethod
    def calculate_angle(
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        point3: Tuple[float, float]
    ) -> float:
        """
        Calculate angle between three points (point2 is the vertex).
        
        Args:
            point1: First point (x, y)
            point2: Vertex point (x, y)
            point3: Third point (x, y)
            
        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)
        
        # Create vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        
        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        
        return angle
    
    @staticmethod
    def calculate_rotation_angle(
        left_point: Tuple[float, float],
        right_point: Tuple[float, float],
        reference_vertical: bool = False
    ) -> float:
        """
        Calculate rotation angle of a line segment.
        
        Args:
            left_point: Left point (x, y)
            right_point: Right point (x, y)
            reference_vertical: If True, measure from vertical axis, else from horizontal
            
        Returns:
            Rotation angle in degrees
        """
        dx = right_point[0] - left_point[0]
        dy = right_point[1] - left_point[1]
        
        # Calculate angle from horizontal
        angle = math.atan2(dy, dx) * 180.0 / np.pi
        
        if reference_vertical:
            # Convert to angle from vertical
            angle = 90.0 - angle
        
        return angle
    
    @staticmethod
    def golf_swing_angles(landmarks: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Calculate all relevant angles for golf swing analysis.
        
        Args:
            landmarks: Dictionary of landmark positions with visibility
            
        Returns:
            Dictionary of calculated angles
        """
        if not landmarks or len(landmarks) < 3:
            return {}
        
        angles = {}
        
        # Extract just x, y coordinates (ignore visibility for angle calculation)
        def get_point(name: str) -> Optional[Tuple[float, float]]:
            if name in landmarks:
                return (landmarks[name][0], landmarks[name][1])
            return None
        
        # 1. Left Elbow Angle (arm bend)
        left_shoulder = get_point('left_shoulder')
        left_elbow = get_point('left_elbow')
        left_wrist = get_point('left_wrist')
        
        if all([left_shoulder, left_elbow, left_wrist]):
            angles['left_elbow'] = AngleCalculator.calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
        
        # 2. Right Elbow Angle (arm bend)
        right_shoulder = get_point('right_shoulder')
        right_elbow = get_point('right_elbow')
        right_wrist = get_point('right_wrist')
        
        if all([right_shoulder, right_elbow, right_wrist]):
            angles['right_elbow'] = AngleCalculator.calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )
        
        # 3. Left Knee Angle (leg bend)
        left_hip = get_point('left_hip')
        left_knee = get_point('left_knee')
        left_ankle = get_point('left_ankle')
        
        if all([left_hip, left_knee, left_ankle]):
            angles['left_knee'] = AngleCalculator.calculate_angle(
                left_hip, left_knee, left_ankle
            )
        
        # 4. Right Knee Angle (leg bend)
        right_hip = get_point('right_hip')
        right_knee = get_point('right_knee')
        right_ankle = get_point('right_ankle')
        
        if all([right_hip, right_knee, right_ankle]):
            angles['right_knee'] = AngleCalculator.calculate_angle(
                right_hip, right_knee, right_ankle
            )
        
        # 5. Shoulder Rotation (horizontal line through shoulders)
        if left_shoulder and right_shoulder:
            angles['shoulder_rotation'] = AngleCalculator.calculate_rotation_angle(
                left_shoulder, right_shoulder, reference_vertical=False
            )
        
        # 6. Hip Rotation (horizontal line through hips)
        if left_hip and right_hip:
            angles['hip_rotation'] = AngleCalculator.calculate_rotation_angle(
                left_hip, right_hip, reference_vertical=False
            )
        
        # 7. Spine Angle (from hips to shoulders)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Calculate midpoints
            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )
            hip_mid = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )
            
            # Calculate spine tilt from vertical
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            
            angles['spine_tilt'] = abs(math.atan2(dx, dy) * 180.0 / np.pi)
        
        # 8. Left Shoulder Angle (shoulder-elbow-hip)
        if left_elbow and left_shoulder and left_hip:
            angles['left_shoulder_angle'] = AngleCalculator.calculate_angle(
                left_elbow, left_shoulder, left_hip
            )
        
        # 9. Right Shoulder Angle (shoulder-elbow-hip)
        if right_elbow and right_shoulder and right_hip:
            angles['right_shoulder_angle'] = AngleCalculator.calculate_angle(
                right_elbow, right_shoulder, right_hip
            )
        
        # 10. Left Hip Angle (hip-knee-shoulder)
        if left_shoulder and left_hip and left_knee:
            angles['left_hip_angle'] = AngleCalculator.calculate_angle(
                left_shoulder, left_hip, left_knee
            )
        
        # 11. Right Hip Angle (hip-knee-shoulder)
        if right_shoulder and right_hip and right_knee:
            angles['right_hip_angle'] = AngleCalculator.calculate_angle(
                right_shoulder, right_hip, right_knee
            )
        
        return angles
    
    @staticmethod
    def calculate_angle_velocity(
        angles_over_time: List[Dict[str, float]],
        angle_name: str,
        fps: float = 30.0
    ) -> List[float]:
        """
        Calculate velocity (rate of change) of an angle over time.
        
        Args:
            angles_over_time: List of angle dictionaries for each frame
            angle_name: Name of the angle to track
            fps: Frames per second of the video
            
        Returns:
            List of velocities (degrees per second)
        """
        angles = [frame.get(angle_name, 0) for frame in angles_over_time]
        velocities = []
        
        for i in range(len(angles)):
            if i == 0:
                velocities.append(0)
            else:
                # Calculate change in angle
                delta_angle = angles[i] - angles[i-1]
                # Convert to degrees per second
                velocity = delta_angle * fps
                velocities.append(velocity)
        
        return velocities
    
    @staticmethod
    def smooth_angles(
        angles_over_time: List[Dict[str, float]],
        window_size: int = 5
    ) -> List[Dict[str, float]]:
        """
        Smooth angle measurements using moving average to reduce noise.
        
        Args:
            angles_over_time: List of angle dictionaries for each frame
            window_size: Size of moving average window
            
        Returns:
            Smoothed angle dictionaries
        """
        if len(angles_over_time) < window_size:
            return angles_over_time
        
        # Get all angle names
        angle_names = set()
        for frame_angles in angles_over_time:
            angle_names.update(frame_angles.keys())
        
        smoothed = []
        
        for i in range(len(angles_over_time)):
            smoothed_frame = {}
            
            for angle_name in angle_names:
                # Get window of angles
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(angles_over_time), i + window_size // 2 + 1)
                
                window_angles = [
                    angles_over_time[j].get(angle_name, None)
                    for j in range(start_idx, end_idx)
                ]
                
                # Filter out None values
                window_angles = [a for a in window_angles if a is not None]
                
                if window_angles:
                    smoothed_frame[angle_name] = np.mean(window_angles)
            
            smoothed.append(smoothed_frame)
        
        return smoothed
    
    @staticmethod
    def get_angle_range(
        angles_over_time: List[Dict[str, float]],
        angle_name: str
    ) -> Tuple[float, float, float]:
        """
        Get min, max, and range of an angle throughout the swing.
        
        Args:
            angles_over_time: List of angle dictionaries
            angle_name: Name of the angle
            
        Returns:
            Tuple of (min, max, range)
        """
        angles = [frame.get(angle_name, None) for frame in angles_over_time]
        angles = [a for a in angles if a is not None]
        
        if not angles:
            return (0.0, 0.0, 0.0)
        
        min_angle = min(angles)
        max_angle = max(angles)
        angle_range = max_angle - min_angle
        
        return (min_angle, max_angle, angle_range)


class SwingPhaseDetector:
    """
    Detects different phases of golf swing based on angle changes.
    """
    
    @staticmethod
    def detect_phases(
        angles_over_time: List[Dict[str, float]],
        fps: float = 30.0
    ) -> Dict[str, int]:
        """
        Detect key phases of the golf swing.
        
        Phases:
        - Address: Setup position
        - Backswing: Start to top of swing
        - Top: Highest point of backswing
        - Downswing: Top to impact
        - Impact: Ball contact
        - Follow-through: After impact
        
        Args:
            angles_over_time: List of angle dictionaries
            fps: Frames per second
            
        Returns:
            Dictionary mapping phase names to frame indices
        """
        if len(angles_over_time) < 10:
            return {}
        
        # Track hip rotation to identify swing phases
        hip_rotations = [frame.get('hip_rotation', 0) for frame in angles_over_time]
        
        phases = {}
        
        # Address is the first frame
        phases['address'] = 0
        
        # Find top of backswing (maximum hip rotation)
        top_frame = np.argmax(np.abs(hip_rotations))
        phases['top'] = top_frame
        
        # Backswing is midpoint to top
        phases['backswing'] = top_frame // 2
        
        # Find impact (rapid deceleration or minimum after top)
        if top_frame < len(hip_rotations) - 5:
            post_top = hip_rotations[top_frame:]
            # Look for zero crossing or minimum
            impact_offset = 0
            for i in range(1, len(post_top)):
                if abs(post_top[i]) < abs(post_top[i-1]):
                    impact_offset = i
                    break
            
            phases['impact'] = top_frame + impact_offset
            phases['downswing'] = (top_frame + phases['impact']) // 2
            phases['follow_through'] = min(phases['impact'] + 10, len(angles_over_time) - 1)
        
        return phases
