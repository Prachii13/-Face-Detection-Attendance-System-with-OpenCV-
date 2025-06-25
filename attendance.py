import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# Load known faces
path = 'images'
images = []
classNames = []
for filename in os.listdir(path):
    img = cv2.imread(f'{path}/{filename}')
    images.append(face_recognition.load_image_file(f'{path}/{filename}'))
    classNames.append(os.path.splitext(filename)[0])

# Encode faces
encodeListKnown = [face_recognition.face_encodings(img)[0] for img in images]

def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        data = f.read()
        if name not in data:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{name},{now}\n')
            print(f"✅ Marked {name} at {now}")

# Start webcam
cap = cv2.VideoCapture(0)

print("🧍 Starting Face Recognition... Press Q to quit.")

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces_current = face_recognition.face_locations(rgb_small)
    encodings_current = face_recognition.face_encodings(rgb_small, faces_current)

    for encodeFace, faceLoc in zip(encodings_current, faces_current):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        face_distances = face_recognition.face_distance(encodeListKnown, encodeFace)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = classNames[best_match_index].title()
            markAttendance(name)

            y1, x2, y2, x1 = [i * 4 for i in faceLoc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Face Attendance', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
