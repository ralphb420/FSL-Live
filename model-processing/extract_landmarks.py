import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

DATASET_PATH = "dataset"
OUTPUT_CSV = "landmarks.csv"

def process_single_image(args):
    """Process a single image and extract landmarks"""
    image_path, label = args
    
    # Initialize MediaPipe for this process
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        model_complexity=0  # 0=fastest, still very accurate
    )
    
    image = cv2.imread(image_path)
    if image is None:
        hands.close()
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    landmarks = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    # Pad to fixed size for 2 hands = 126 values
    while len(landmarks) < 126:
        landmarks.append(0)
    
    landmarks.append(label)
    hands.close()
    
    return landmarks

if __name__ == '__main__':
    print("Collecting image paths...")
    
    # Collect all image paths and labels
    image_tasks = []
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        
        if not os.path.isdir(label_path):
            continue
        
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image_tasks.append((image_path, label))
    
    total_images = len(image_tasks)
    print(f"Found {total_images} images to process")
    
    # Determine number of processes
    num_processes = cpu_count()
    print(f"Using {num_processes} CPU cores")
    
    start_time = time.time()
    
    # Process images in parallel with progress bar
    print("Processing images...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, image_tasks),
            total=total_images,
            desc="Extracting landmarks"
        ))
    
    # Filter out None results (failed images)
    data = [r for r in results if r is not None]
    
    elapsed_time = time.time() - start_time
    
    print(f"\nProcessed {len(data)}/{total_images} images successfully")
    print(f"Time taken: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print(f"Speed: {len(data)/elapsed_time:.2f} images/second")
    
    # Save to CSV
    print(f"Saving to {OUTPUT_CSV}...")
    columns = [f"f{i}" for i in range(126)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("✓ Landmark extraction complete!")
    print(f"✓ Saved to {OUTPUT_CSV}")