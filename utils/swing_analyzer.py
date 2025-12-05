"""
Swing Analyzer Module

Analyzes golf swing based on pose landmarks and angles.
Provides scoring and feedback based on biomechanical principles.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SwingScore:
    """Data class to store swing analysis scores."""
    overall_score: float
    posture_score: float
    rotation_score: float
    tempo_score: float
    balance_score: float
    feedback: List[str]
    metrics: Dict[str, float]


class SwingAnalyzer:
    """
    Analyzes golf swing and provides scoring based on biomechanical principles.
    """
    
    # Ideal angle ranges based on professional golf biomechanics
    IDEAL_RANGES = {
        'spine_tilt': (25, 45),  # degrees from vertical at address
        'left_knee': (140, 170),  # slight flex
        'right_knee': (140, 170),  # slight flex
        'hip_rotation': (45, 90),  # max hip turn in backswing
        'shoulder_rotation': (90, 110),  # max shoulder turn
        'left_elbow': (140, 180),  # lead arm should stay straight
        'right_elbow': (90, 140),  # trail arm bent in backswing
    }
    
    def __init__(self):
        """Initialize the swing analyzer."""
        pass
    
    def analyze_swing(
        self,
        angles_over_time: List[Dict[str, float]],
        landmarks_over_time: List[Dict],
        fps: float = 30.0
    ) -> SwingScore:
        """
        Comprehensive swing analysis.
        
        Args:
            angles_over_time: List of angle dictionaries for each frame
            landmarks_over_time: List of landmark dictionaries for each frame
            fps: Frames per second
            
        Returns:
            SwingScore object with detailed analysis
        """
        if not angles_over_time or len(angles_over_time) < 10:
            return SwingScore(
                overall_score=0.0,
                posture_score=0.0,
                rotation_score=0.0,
                tempo_score=0.0,
                balance_score=0.0,
                feedback=["Insufficient data for analysis"],
                metrics={}
            )
        
        # Calculate individual component scores
        posture_score = self._analyze_posture(angles_over_time)
        rotation_score = self._analyze_rotation(angles_over_time)
        tempo_score = self._analyze_tempo(angles_over_time, fps)
        balance_score = self._analyze_balance(landmarks_over_time)
        
        # Calculate overall score (weighted average)
        overall_score = (
            posture_score * 0.25 +
            rotation_score * 0.25 +
            tempo_score * 0.25 +
            balance_score * 0.25
        )
        
        # Generate feedback
        feedback = self._generate_feedback(
            posture_score,
            rotation_score,
            tempo_score,
            balance_score,
            angles_over_time
        )
        
        # Collect key metrics
        metrics = self._extract_metrics(angles_over_time)
        
        return SwingScore(
            overall_score=round(overall_score, 1),
            posture_score=round(posture_score, 1),
            rotation_score=round(rotation_score, 1),
            tempo_score=round(tempo_score, 1),
            balance_score=round(balance_score, 1),
            feedback=feedback,
            metrics=metrics
        )
    
    def _analyze_posture(self, angles_over_time: List[Dict[str, float]]) -> float:
        """
        Analyze posture quality.
        
        Checks:
        - Spine angle at address
        - Knee bend consistency
        - Upper body tilt
        """
        score = 100.0
        deductions = []
        
        # Check address position (first few frames)
        address_frames = angles_over_time[:min(5, len(angles_over_time))]
        
        # Spine tilt
        spine_tilts = [f.get('spine_tilt', 0) for f in address_frames if 'spine_tilt' in f]
        if spine_tilts:
            avg_spine_tilt = np.mean(spine_tilts)
            if not (self.IDEAL_RANGES['spine_tilt'][0] <= avg_spine_tilt <= self.IDEAL_RANGES['spine_tilt'][1]):
                score -= 15
                deductions.append(f"Spine tilt {avg_spine_tilt:.1f}° is outside ideal range")
        
        # Knee bend
        left_knees = [f.get('left_knee', 180) for f in address_frames if 'left_knee' in f]
        right_knees = [f.get('right_knee', 180) for f in address_frames if 'right_knee' in f]
        
        if left_knees:
            avg_left_knee = np.mean(left_knees)
            if avg_left_knee > 175:  # Too straight
                score -= 10
                deductions.append("Left knee too straight")
            elif avg_left_knee < 130:  # Too bent
                score -= 10
                deductions.append("Left knee bent too much")
        
        if right_knees:
            avg_right_knee = np.mean(right_knees)
            if avg_right_knee > 175:
                score -= 10
                deductions.append("Right knee too straight")
            elif avg_right_knee < 130:
                score -= 10
                deductions.append("Right knee bent too much")
        
        return max(0, score)
    
    def _analyze_rotation(self, angles_over_time: List[Dict[str, float]]) -> float:
        """
        Analyze rotation quality.
        
        Checks:
        - Hip rotation range
        - Shoulder rotation range
        - Shoulder-hip separation (X-factor)
        """
        score = 100.0
        
        # Get max hip rotation
        hip_rotations = [abs(f.get('hip_rotation', 0)) for f in angles_over_time]
        max_hip_rotation = max(hip_rotations) if hip_rotations else 0
        
        # Get max shoulder rotation
        shoulder_rotations = [abs(f.get('shoulder_rotation', 0)) for f in angles_over_time]
        max_shoulder_rotation = max(shoulder_rotations) if shoulder_rotations else 0
        
        # Check hip rotation
        if max_hip_rotation < 30:
            score -= 20
        elif max_hip_rotation < 45:
            score -= 10
        elif max_hip_rotation > 100:
            score -= 5  # Too much rotation can be unstable
        
        # Check shoulder rotation
        if max_shoulder_rotation < 60:
            score -= 20
        elif max_shoulder_rotation < 80:
            score -= 10
        
        # X-factor (shoulder-hip separation)
        x_factor = max_shoulder_rotation - max_hip_rotation
        if x_factor < 10:
            score -= 15  # Not enough separation
        elif x_factor > 50:
            score -= 10  # Too much tension
        
        return max(0, score)
    
    def _analyze_tempo(self, angles_over_time: List[Dict[str, float]], fps: float) -> float:
        """
        Analyze swing tempo and smoothness.
        
        Checks:
        - Consistent speed
        - No jerky movements
        - Proper timing
        """
        score = 100.0
        
        if len(angles_over_time) < 10:
            return 50.0
        
        # Analyze hip rotation velocity
        hip_rotations = [f.get('hip_rotation', 0) for f in angles_over_time]
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(hip_rotations)):
            velocity = abs(hip_rotations[i] - hip_rotations[i-1]) * fps
            velocities.append(velocity)
        
        if velocities:
            # Check for smoothness (low variance = smooth)
            velocity_std = np.std(velocities)
            
            # Penalize jerky movements
            if velocity_std > 50:
                score -= 20
            elif velocity_std > 30:
                score -= 10
            
            # Check swing duration (ideal: 1-1.5 seconds)
            duration = len(angles_over_time) / fps
            if duration < 0.8:
                score -= 15  # Too fast
            elif duration > 2.0:
                score -= 10  # Too slow
        
        return max(0, score)
    
    def _analyze_balance(self, landmarks_over_time: List[Dict]) -> float:
        """
        Analyze balance throughout the swing.
        
        Checks:
        - Head movement (should stay relatively still)
        - Weight distribution
        - Stable base
        """
        score = 100.0
        
        if not landmarks_over_time or len(landmarks_over_time) < 5:
            return 50.0
        
        # Track head movement (nose position)
        nose_positions = []
        for landmarks in landmarks_over_time:
            if 'nose' in landmarks:
                nose_positions.append(landmarks['nose'][:2])  # x, y only
        
        if len(nose_positions) >= 5:
            # Calculate head movement variance
            x_positions = [p[0] for p in nose_positions]
            y_positions = [p[1] for p in nose_positions]
            
            x_variance = np.std(x_positions)
            y_variance = np.std(y_positions)
            
            # Excessive head movement is bad
            if x_variance > 50:  # pixels
                score -= 15
            if y_variance > 50:
                score -= 15
        
        # Check stance width (distance between ankles)
        ankle_widths = []
        for landmarks in landmarks_over_time[:10]:  # Check address position
            if 'left_ankle' in landmarks and 'right_ankle' in landmarks:
                left = landmarks['left_ankle'][:2]
                right = landmarks['right_ankle'][:2]
                width = abs(left[0] - right[0])
                ankle_widths.append(width)
        
        if ankle_widths:
            avg_width = np.mean(ankle_widths)
            # Stance should be shoulder-width or slightly wider
            if avg_width < 50:  # Too narrow
                score -= 10
        
        return max(0, score)
    
    def _generate_feedback(
        self,
        posture_score: float,
        rotation_score: float,
        tempo_score: float,
        balance_score: float,
        angles_over_time: List[Dict[str, float]]
    ) -> List[str]:
        """Generate actionable feedback based on scores."""
        feedback = []
        
        # Overall assessment
        overall = (posture_score + rotation_score + tempo_score + balance_score) / 4
        
        if overall >= 90:
            feedback.append("🎯 Excellent swing! Very close to professional form.")
        elif overall >= 75:
            feedback.append("👍 Good swing! Some minor improvements possible.")
        elif overall >= 60:
            feedback.append("⚠️ Decent swing with room for improvement.")
        else:
            feedback.append("❌ Needs significant work. Consider lessons.")
        
        # Specific feedback
        if posture_score < 70:
            feedback.append("📐 Posture: Work on spine angle and knee flex at address.")
        
        if rotation_score < 70:
            hip_rots = [abs(f.get('hip_rotation', 0)) for f in angles_over_time]
            max_hip = max(hip_rots) if hip_rots else 0
            if max_hip < 45:
                feedback.append("🔄 Rotation: Increase hip turn in backswing.")
            feedback.append("🔄 Rotation: Focus on shoulder-hip separation (X-factor).")
        
        if tempo_score < 70:
            feedback.append("⏱️ Tempo: Work on smoother, more controlled swing speed.")
        
        if balance_score < 70:
            feedback.append("⚖️ Balance: Keep head still and maintain stable base.")
        
        # Positive reinforcement for strengths
        if posture_score >= 85:
            feedback.append("✅ Strong: Excellent posture!")
        if rotation_score >= 85:
            feedback.append("✅ Strong: Great rotation mechanics!")
        if tempo_score >= 85:
            feedback.append("✅ Strong: Smooth tempo!")
        if balance_score >= 85:
            feedback.append("✅ Strong: Excellent balance!")
        
        return feedback
    
    def _extract_metrics(self, angles_over_time: List[Dict[str, float]]) -> Dict[str, float]:
        """Extract key metrics for display."""
        metrics = {}
        
        # Max rotations
        hip_rotations = [abs(f.get('hip_rotation', 0)) for f in angles_over_time]
        shoulder_rotations = [abs(f.get('shoulder_rotation', 0)) for f in angles_over_time]
        
        if hip_rotations:
            metrics['max_hip_rotation'] = round(max(hip_rotations), 1)
        if shoulder_rotations:
            metrics['max_shoulder_rotation'] = round(max(shoulder_rotations), 1)
        
        # X-factor
        if hip_rotations and shoulder_rotations:
            metrics['x_factor'] = round(max(shoulder_rotations) - max(hip_rotations), 1)
        
        # Average knee bend
        left_knees = [f.get('left_knee', 180) for f in angles_over_time if 'left_knee' in f]
        right_knees = [f.get('right_knee', 180) for f in angles_over_time if 'right_knee' in f]
        
        if left_knees:
            metrics['avg_left_knee_bend'] = round(180 - np.mean(left_knees), 1)
        if right_knees:
            metrics['avg_right_knee_bend'] = round(180 - np.mean(right_knees), 1)
        
        # Spine tilt at address
        spine_tilts = [f.get('spine_tilt', 0) for f in angles_over_time[:5] if 'spine_tilt' in f]
        if spine_tilts:
            metrics['address_spine_tilt'] = round(np.mean(spine_tilts), 1)
        
        return metrics
