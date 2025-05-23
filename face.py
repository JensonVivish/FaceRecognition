import cv2
import face_recognition
import pickle
import time


def recognize_faces(frame, known_encodings, known_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin() if face_distances.size > 0 else -1

        if best_match_index != -1 and matches[best_match_index]:
            name = known_names[best_match_index]
        face_names.append(name)

    return face_locations, face_names


if __name__ == "__main__":
    encodings_file = "encodings.pkl"

    print("[INFO] Loading face encodings...")
    with open(encodings_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"[DONE] Loaded {len(known_face_encodings)} encodings.")

    video = cv2.VideoCapture(0)
    print("[INFO] Webcam started. Press 'q' to quit.")

    prev_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
