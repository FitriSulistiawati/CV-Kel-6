import cv2

# Load Haarcascade classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
eyeglass_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
def detect_features(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face and label it as "face"
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(vid, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Region of interest (ROI) for detecting eyes, smile, and eyeglasses within the face
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_color = vid[y:y + h, x:x + w]

        # Detect eyes within face region
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 22)
        eyeglasses = eyeglass_classifier.detectMultiScale(roi_gray, 1.1, 22)

        # If eyes are detected, label as "eye"
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)  # Blue for eyes
                cv2.putText(roi_color, 'Eye', (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif len(eyeglasses) > 0:
            # If eyeglasses are detected, label as "eyeglasses"
            for (ex, ey, ew, eh) in eyeglasses:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)  # Yellow for eyeglasses
                cv2.putText(roi_color, 'Eyeglasses', (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # If no eyes or eyeglasses detected, assume no glasses
            cv2.putText(roi_color, 'with Eyeglasses', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Detect smile within face region
        smiles = smile_classifier.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)  # Red for smile
            cv2.putText(roi_color, 'Smile', (sx, sy + sh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return vid
video_capture = cv2.VideoCapture(0)
while True:
    result, video_frame = video_capture.read()  # Read frames from the video
    if not result:
        break  # Terminate the loop if frame is not read successfully
    
    # Apply the feature detection function
    processed_frame = detect_features(video_frame)
    
    # Display the processed frame with detected face, eyes, smile, and eyeglasses
    cv2.imshow("Face, Eye, Smile, and Eyeglasses Detection", processed_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

