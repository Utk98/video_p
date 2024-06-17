import os
import pickle
import face_recognition
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import cv2
import time

# Set up directories and initialize the dictionary to hold face encodings
people_dir = "C:/Users/UTKARSH SINGH/Desktop/video_po/Multiple-Face-Recognition-in-Videos-master/People/"
people = dict()

# Iterate over each person in the people directory to detect and encode faces
for person in os.listdir(people_dir):
    print(person)
    person_dir = os.path.join(people_dir, person)
    person_faces = list()  # List of detected faces for the person
    print("Number of photos:", len(os.listdir(person_dir)))
    print(os.listdir(person_dir))
    
    # Iterate over each photo in the person's directory
    for photo in os.listdir(person_dir):
        if photo.endswith(".db"):
            continue
        photo_path = os.path.join(person_dir, photo)
        image = face_recognition.load_image_file(photo_path)
        face_encodings = face_recognition.face_encodings(image)
        
        # If faces are found, add the encoding to the list
        if len(face_encodings) > 0:
            person_faces.append(face_encodings[0])
    
    name = person
    if len(person_faces) == 0:
        print("No faces were found.")
    else:
        print("Detected Faces from", person, ":", len(person_faces))
        people[name] = person_faces

# Pickling (serializing) the dictionary into a file
with open('encoded_people.pickle', 'wb') as filename:
    pickle.dump(people, filename)

# Load the font
font = PIL.ImageFont.truetype("timesbd.ttf", 20)

# Load the encoded people data
with open('encoded_people.pickle', 'rb') as filename:
    people = pickle.load(filename)
print("Data loaded successfully")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    start_time = time.time()
    scaleFactor = 4

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=1/scaleFactor, fy=1/scaleFactor)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    img_loc = face_recognition.face_locations(rgb_small_frame, model="hog")
    img_enc = face_recognition.face_encodings(rgb_small_frame, known_face_locations=img_loc)

    # Convert frame to PIL image for drawing
    face_img = PIL.Image.fromarray(frame)

    for i in range(0, len(img_enc)):
        for name, encodings in people.items():
            # Compare face encodings
            result = face_recognition.compare_faces(encodings, img_enc[i], tolerance=0.5)
            if True in result:
                print(f"Person detected: {name}")
                # Scale back up face locations since the frame we detected in was scaled to 1/scaleFactor size
                top, right, bottom, left = np.multiply(img_loc[i], scaleFactor)
                draw = PIL.ImageDraw.Draw(face_img)
                # Draw rectangle around the face
                draw.rectangle([left, top, right, bottom], outline="red", width=2)
                # Draw the name of the person
                draw.text((left, bottom), name, font=font, fill="red")

    # Display the resulting frame
    open_cv_image = np.array(face_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    cv2.imshow('frame', open_cv_image)

    print("--- %s seconds ---" % (time.time() - start_time))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
