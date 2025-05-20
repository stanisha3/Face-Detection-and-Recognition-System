import cv2
import face_recognition
import numpy as np

known_image_1 = face_recognition.load_image_file("bhavya.jpg")
known_image_2 = face_recognition.load_image_file("tanisha.jpg")
known_image_3 = face_recognition.load_image_file("mrinal.jpg")
known_image_4 = face_recognition.load_image_file("prashansa.jpg")
known_image_5 = face_recognition.load_image_file("garima.jpg")
known_image_6 = face_recognition.load_image_file("surendra_sir.jpg")

encoding_1 = face_recognition.face_encodings(known_image_1)[0]
encoding_2 = face_recognition.face_encodings(known_image_2)[0]
encoding_3 = face_recognition.face_encodings(known_image_3)[0]
encoding_4 = face_recognition.face_encodings(known_image_4)[0]
encoding_5 = face_recognition.face_encodings(known_image_5)[0]
encoding_6 = face_recognition.face_encodings(known_image_6)[0]

known_encodings = [encoding_1, encoding_2, encoding_3, encoding_4, encoding_5, encoding_6]
known_names = ["Bhavya", "Tanisha", "Mrinal", "Prashansa", "Garima", "Surendra Sir"]

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
