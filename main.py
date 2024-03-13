# Importing necessary libraries/modules
import cv2                     # take input from webcam or any external camera module, process it and give it to face_recognition
import face_recognition        # it will recognise the faces and compare the faces from the already present faces in our database
import csv                     # to handle the csv file, creating and updating
from datetime import datetime  # used to get exact date and time
import os                      # used to access files, you can check if a file or directory exists, create directories, and perform other file system operations.
import numpy as np             # used for calculations


# This line initializes a video capture object using the OpenCV library.
video_capture = cv2.VideoCapture(0)    # we are taking input from default webcam so 0 is written in brackets
# VideoCapture --> Class
# video_capture --> object


# Calling all the photos in our program (that are saved in folder named 'photos')

# It loads the image file named "monalisa.jpg" from the "photos" directory and stores the image data in the variable monalisa_image
monalisa_image = face_recognition.load_image_file("photos/monalisa.jpg")
# this line of code calculates the facial encoding of the Mona Lisa's face
monalisa_encoding = face_recognition.face_encodings(monalisa_image)[0]

sachin_image = face_recognition.load_image_file("photos/sachin.jpg")
sachin_encoding = face_recognition.face_encodings(sachin_image)[0]

tata_image = face_recognition.load_image_file("photos/tata.jpg")
tata_encoding = face_recognition.face_encodings(tata_image)[0]

tesla_image = face_recognition.load_image_file("photos/tesla.jpeg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

# Storing known face encodings and names
# The order of names in the known_faces_names list should correspond to the order of facial encodings in the known_face_encoding list. If the order is not consistent, it would result in mislabeling and marking the attendance of the wrong person.
known_face_encoding = [monalisa_encoding, sachin_encoding, tata_encoding, tesla_encoding]
known_faces_names = ["MonaLisa", "Sachin", "Ratan Tata", "Tesla"]


# Creating a list of students (initially the same as known_faces_names)
students = known_faces_names.copy()

# These variables will be used for the face that is coming from the webcam
face_locations = []   # will be used to save the face location if there is a face in the frame that is coming from the video capture (face coordinates)
face_encodings = []   # The raw data
face_names = []
s = True

# Getting the current date
now = datetime.now()  # datetime is a module, `now()` is a function
today = now.strftime("%Y-%m-%d")

# Creating a csv file
f = open(today+'.csv', 'w+', newline = '')    # w+ method is used to do both write and read a file
lnwriter = csv.writer(f)

# Define a threshold value for face distance
threshold = 0.6  # Adjust this value as needed


# Infinite loop for capturing video frames and performing face recognition
while True:

    # The `read()` method of the `VideoCapture` object returns a tuple where the first element is a boolean value indicating whether the frame was successfully captured (`True` for success, `False` otherwise), and the second element is the actual frame.
    # The underscore `_` is a convention in Python to indicate that the first element of the tuple is intentionally ignored. In this case, it's the success flag, and the frame is assigned to the variable `frame`
    _, frame = video_capture.read()  # This line captures a single frame from the video source specified by the `video_capture` object

    # `fx=0.25`: This is the scaling factor for the width of the frame.It reduces the width to 25% of the original
    # `fy=0.25`: This is the scaling factor for the height of the frame.It reduces the height to 25% of the original
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)  # decreasing the size of the input that is coming from the webcam

    rgb_small_frame = small_frame[:,:,::-1]  # converting small_frame to rgb format because cv2 takes input in the bgr format

    if s:
        # detects the locations of faces in the input image and stores the bounding box coordinates of the detected faces in the face_locations variable
        face_locations = face_recognition.face_locations(rgb_small_frame)  # face_locations() function returns a list of tuples, where each tuple is the bounding box coordinates of a face
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # this line of code computes the facial encodings for the faces detected in the input image
        face_names = []

        # Matching the faces with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)  # function returns a list of boolean values indicating whether the current face encoding matches any of the known encodings. If a match is found, the corresponding boolean value is True; otherwise, it is False.
            # matches is a list
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)  # face_distance is a numerical value representing the distance (or dissimilarity) between the two face encodings. Smaller distances indicate greater similarity between the faces.

            # Filter out potential matches below the threshold
            if np.min(face_distance) <= threshold:
                best_match_index = np.argmin(face_distance)  # the input array (for `argmin()` function) is `face_distance`, which contains numerical values representing the distances (or dissimilarities) between the current face encoding and each known face encoding.
                # `best_match_index` holds the index of the smallest value in `face_distance` array. That smallest value has the closest match to the current face encoding

                if matches[best_match_index]:  # goes to the matches list and checks whether value at the best_match_index is true or not
                    name = known_faces_names[best_match_index]  # if true then goes to known_faces_names list and fetches name at the best_match_index and assigns name of the individual to the variable `name`

            # the `name` variable will become empty after each iteration, so a `face_name` list is created to record name of present individuals
            face_names.append(name)

            # Updating attendance and removing the student from the list if recognized
            if name in known_faces_names:
                if name in students:
                    students.remove(name)  # removing name from students list so that someone's attendance does not get marked again(if that individual stands in front of the camera again)
                    print(students)  # this is just to check during run time that the program is identifying the correct individual
                    current_time = now.strftime("%H-%M-%S")  # retrieves the current time in the format "hour-minute-second"
                    lnwriter.writerow([name, current_time])  # writes a new row to the CSV file (lnwriter) containing the recognized name and the current time.

    # Displaying the frame with recognized faces
    cv2.imshow("Attendance System", frame)  # shows live video feed window

    # Breaking the loop if 'q' is pressed
    if (cv2.waitKey(1) & 0xFF == ord('q')):  # waitKey function returns the ASCII code of the pressed key and `-1` if no key is pressed
        # `& 0xFF` masks the ASCII code returned by cv2.waitKey(1), ensuring that only the least significant byte (LSB) remains.
        # `== ord('q')` compares the masked ASCII code with the ASCII code of the 'q' key.
        break

# This block of code is responsible for releasing the resources used for video capture, closing any open windows, and closing the CSV file.
video_capture.release()  # This line releases the video capture object created earlier (`video_capture`) and frees up any resources associated with it. Releasing the video capture is important to ensure that the camera or video source is properly closed
cv2.destroyAllWindows()  # This line closes all OpenCV windows that were opened during the program's execution
f.close()  # This line closes the CSV file (f) that was opened earlier for writing attendance data.
