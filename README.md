
# 🧠 Face Recognition & Tracking System

Welcome to our Face Recognition Project — a real-time computer vision application built using Python, OpenCV, face_recognition, and Deep SORT (because one tracker is never enough). 😎

This README will walk you through the whole process — from encoding faces to running the system like a pro (or at least pretending to be one).

---

## ⚙️ What It Does

- Encodes known faces from a folder (`known_faces/`)
- Detects and recognizes faces from live webcam feed
- Tracks them using Deep SORT (so the system doesn't get confused when people move around)
- Draws a box around each face, labels them, and shows live FPS
- Absolutely does *not* call the FBI... unless you add that part 😉

---

## 📁 Folder Structure


project/
│
├── known_faces/           # Add known person images here (JPEG/PNG only please)
├── face.py    # Main execution file
├── encode_faces.py        # Script to encode faces and save to .pkl
├── encodings.pkl          # Stores encoded face data
├── README.md              # You’re reading this!

## Requirements

setuptools>=65.0.0
opencv-python>=4.5.5
face_recognition>=1.3.0
dlib>=19.22
numpy>=1.19
scikit-learn>=0.24
scipy>=1.5
imutils>=0.5
deep_sort_realtime>=1.3
filterpy>=1.4

<pre>  pip install setuptools>=65.0.0 opencv-python>=4.5.5 face_recognition>=1.3.0 dlib>=19.22 numpy>=1.19 scikit-learn>=0.24 scipy>=1.5 imutils>=0.5 deep_sort_realtime>=1.3 filterpy>=1.4 </pre>

