import cv2 
import depthai as dai
import numpy as np
import face_recognition
#create a pipeline
pipeline = dai.Pipeline()

#define the source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

#properties

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(860, 720)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

#Linking
camRgb.video.link(xoutVideo.input)

abbaas_image = face_recognition.load_image_file("ali_picture.jpg")
abbaas_face_encoding = face_recognition.face_encodings(abbaas_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    abbaas_face_encoding,
]
known_face_names = [
    "Alireza"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
top_lip = []
bottom_lip=[]
center_points = []
process_this_frame = True

with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    while True:
        videoIn = video.get()
        
        #Get BGR from NV12 encoded video frame to show with opencv
        frame = videoIn.getCvFrame()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            face_names = []
            for index, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if name == 'Alireza':
                    keys = list(face_landmarks_list[index].keys())
                    top_lip = face_landmarks_list[index][keys[-2]]
                    bottom_lip = face_landmarks_list[index][keys[-1]]
                    top_lip = np.array(top_lip, dtype=np.int32)
                    bottom_lip = np.array(bottom_lip, dtype=np.int32)
                    top_lip = top_lip*4
                    bottom_lip = bottom_lip*4
                    center_top_lip = np.mean(top_lip, axis=0)
                    center_top_lip = center_top_lip.astype('int')
                    center_points.append(center_top_lip)
                # print(face_landmarks_list[index][keys[-2]])
                face_names.append(name)
        process_this_frame = not process_this_frame


        # Display the results
        cv2.polylines(frame, np.array([top_lip]), 1, (255,255,255))
        cv2.polylines(frame,np.array([bottom_lip]), 1, (255,255,255))
        for i in range(1, len(center_points)):
            if center_points[i-1] is None or center_points[i] is None:
                continue
            cv2.line(frame, tuple(center_points[i-1]), tuple(center_points[i]), (0,0,255), 2)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        #Display the resulting image
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()