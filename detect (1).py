# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time

def initialize_models():
    """Initialize all models with proper error handling"""
    print("Initializing models...")
    models = {}
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Verify models directory exists
    if not os.path.exists(models_dir):
        print(f"Error: 'models' directory not found at {models_dir}")
        print("Please create a 'models' folder and add all required model files")
        return None

    # Face detection model
    try:
        face_proto = os.path.join(models_dir, "opencv_face_detector.pbtxt")
        face_model = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
        
        if not os.path.exists(face_proto):
            print(f"Missing file: {face_proto}")
        if not os.path.exists(face_model):
            print(f"Missing file: {face_model}")
        
        models['face_net'] = cv2.dnn.readNet(face_model, face_proto)
        print("✓ Face detection model loaded")
    except Exception as e:
        print(f"× Face detection model error: {e}")
        try:
            models['haar_cascade'] = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✓ Haar cascade loaded as fallback")
        except Exception as e:
            print(f"× Haar cascade error: {e}")
            return None

    # Gender model
    try:
        gender_proto = os.path.join(models_dir, "gender_deploy.prototxt")
        gender_model = os.path.join(models_dir, "gender_net.caffemodel")
        models['gender_net'] = cv2.dnn.readNet(gender_model, gender_proto)
        print("✓ Gender model loaded")
    except Exception as e:
        print(f"× Gender model error: {e}")

    # Age model
    try:
        age_proto = os.path.join(models_dir, "age_deploy.prototxt")
        age_model = os.path.join(models_dir, "age_net.caffemodel")
        models['age_net'] = cv2.dnn.readNet(age_model, age_proto)
        print("✓ Age model loaded")
    except Exception as e:
        print(f"× Age model error: {e}")

    return models

def detect_faces(models, frame):
    """Detect faces using available models"""
    face_boxes = []
    
    # Try DNN model first
    if 'face_net' in models:
        try:
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
            models['face_net'].setInput(blob)
            detections = models['face_net'].forward()
            
            h, w = frame.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    face_boxes.append(box.astype("int"))
        except Exception as e:
            print(f"Face detection error: {e}")

    # Fallback to Haar cascade if no faces detected
    if not face_boxes and 'haar_cascade' in models:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = models['haar_cascade'].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_boxes = [[x, y, x+w, y+h] for (x, y, w, h) in faces]
        except Exception as e:
            print(f"Haar cascade error: {e}")

    return face_boxes

def predict_age_gender(models, face_roi):
    """Predict age and gender for a face region"""
    gender = "Unknown"
    age = "Unknown"
    
    if 'gender_net' in models or 'age_net' in models:
        try:
            blob = cv2.dnn.blobFromImage(
                face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), 
                swapRB=False)
            
            # Predict gender
            if 'gender_net' in models:
                models['gender_net'].setInput(blob)
                gender_preds = models['gender_net'].forward()
                gender = "Male" if gender_preds[0].argmax() == 0 else "Female"
            
            # Predict age
            if 'age_net' in models:
                models['age_net'].setInput(blob)
                age_preds = models['age_net'].forward()
                age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                age = age_list[age_preds[0].argmax()]
        except Exception as e:
            print(f"Age/gender prediction error: {e}")
    
    return gender, age

def main():
    # Initialize models
    models = initialize_models()
    if not models:
        print("Failed to initialize required models")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        face_boxes = detect_faces(models, frame)
        
        # Process each face
        for box in face_boxes:
            x1, y1, x2, y2 = box
            
            # Add padding
            padding = 20
            x1, y1 = max(0, x1-padding), max(0, y1-padding)
            x2, y2 = min(frame.shape[1], x2+padding), min(frame.shape[0], y2+padding)
            
            # Extract face ROI
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            
            # Predict age and gender
            gender, age = predict_age_gender(models, face_roi)
            
            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Age and Gender Detection", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()