import os
import face_recognition
import pickle

KNOWN_FACES_DIR = "known_faces"
encodings_file = "encodings.pkl"

known_face_encodings = []
known_face_names = []

print("[INFO] Building face encodings cache...")

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        filepath = os.path.join(person_folder, filename)

        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(person_name)
            print(f"[ENCODED] {filename}")
        else:
            print(f"[SKIPPED] No face found in {filename}")

# Save encodings to file
with open(encodings_file, "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"[DONE] Encodings saved to {encodings_file}")
