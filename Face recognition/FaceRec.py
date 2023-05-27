import cv2

# Load the trained XML classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open a video capture object to capture frames from the camera (index 0)
cap = cv2.VideoCapture(0)

# Create an LBPH face recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the saved face data
recognizer.read('face_data.xml')

# Main loop for capturing and processing frames
while True:
    # Read a frame from the camera
    ret, img = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the face cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face on the original frame
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Extract the region of interest (ROI) for face recognition
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Perform face recognition
        label, confidence = recognizer.predict(roi_gray)

        # Determine the label text based on confidence level
        if confidence < 100:
            label_text = "Me"
        else:
            label_text = "Unknown"

        # Display the label on the face rectangle
        cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Display the resulting frame in a window named 'img'
    cv2.imshow('img', img)

    # Wait for the 'Esc' key to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
