import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

def extract_landmarks_from_video(video_path, max_frames=64):
    """+
    Extract hand, pose, and face landmarks from video
    Returns: numpy array of shape (num_frames, num_features)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    landmarks_sequence = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = holistic.process(frame_rgb)
            
            # Extract landmarks
            frame_landmarks = []
            
            # Left hand (21 landmarks x 3 coords = 63 features)
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_landmarks.extend([0.0] * 63)  # Fill with zeros if not detected
            
            # Right hand (21 landmarks x 3 coords = 63 features)
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_landmarks.extend([0.0] * 63)
            
            # Pose (33 landmarks x 3 coords = 99 features) - upper body only
            if results.pose_landmarks:
                # Only use upper body landmarks (0-10: face/shoulders/arms)
                for i in range(11):  # 0-10 landmarks
                    landmark = results.pose_landmarks.landmark[i]
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_landmarks.extend([0.0] * 33)
            
            landmarks_sequence.append(frame_landmarks)
            frame_count += 1
    
    cap.release()
    
    if len(landmarks_sequence) == 0:
        return None
    
    return np.array(landmarks_sequence, dtype=np.float32)


def pad_sequence(sequence, max_length=64):
    """Pad or truncate sequence to fixed length"""
    if sequence is None:
        return np.zeros((max_length, 159), dtype=np.float32)  # 63+63+33 features
    
    seq_len = len(sequence)
    
    if seq_len >= max_length:
        # Truncate
        return sequence[:max_length]
    else:
        # Pad with last frame
        padding = np.tile(sequence[-1], (max_length - seq_len, 1))
        return np.vstack([sequence, padding])


def extract_all_features(json_path, video_root, output_dir, max_frames=64, max_videos=None):
    """
    Extract features from all videos in dataset
    max_videos: If set, only process first N videos (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    features_dict = {}
    skipped = 0
    
    # Limit videos if testing
    video_ids = list(data.keys())
    if max_videos:
        video_ids = video_ids[:max_videos]
        print(f"⚠️  TEST MODE: Processing only {max_videos} videos")
    
    print(f"Extracting landmarks from {len(video_ids)} videos...")
    
    for vid in tqdm(video_ids, desc="Processing videos"):
        video_path = os.path.join(video_root, vid + '.mp4')
        
        if not os.path.exists(video_path):
            skipped += 1
            continue
        
        # Extract landmarks
        landmarks = extract_landmarks_from_video(video_path, max_frames)
        
        if landmarks is None:
            skipped += 1
            continue
        
        # Pad to fixed length
        landmarks_padded = pad_sequence(landmarks, max_frames)
        
        # Store with metadata
        features_dict[vid] = {
            'features': landmarks_padded,
            'label': data[vid]['action'][0],
            'subset': data[vid]['subset']
        }
    
    # Save features
    output_file = os.path.join(output_dir, 'mediapipe_features.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Processed: {len(features_dict)} videos")
    print(f"   Skipped: {skipped} videos")
    print(f"   Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default="../../../preprocessed/nslt_2000.json")
    parser.add_argument('--video_root', type=str, default="../../../WLASL2000/WLASL2000")
    parser.add_argument('--output_dir', type=str, default="../../../mediapipe_features/")
    parser.add_argument('--max_frames', type=int, default=64)
    args = parser.parse_args()
    
    # Extract features
    print(f"JSON path: {args.json_path}")
    print(f"Video root: {args.video_root}")
    print(f"Output dir: {args.output_dir}")
    
    extract_all_features(args.json_path, args.video_root, args.output_dir, args.max_frames)
