import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import time
import numpy as np
def parse_arguments()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yolov11 live")
    parser.add_argument("--webcam-resolution",default=[1280,720],nargs=2,type=int)
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS,45)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    model = YOLO("../best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1,
        text_color=sv.Color.BLACK,
        text_position=sv.Position.TOP_LEFT
    )

    confidence_threshold = 0.5
    max_box_area_ratio  = 0.6
    pTime = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter by confidence
        detections = detections[detections.confidence > confidence_threshold]

        # Filter by bounding box size (area ratio)
        if len(detections) > 0:
            frame_area = frame.shape[0] * frame.shape[1]
            box_areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            area_ratios = box_areas / frame_area
            size_mask = area_ratios < max_box_area_ratio
            detections = detections[size_mask]

        #  labels with class names and confidence scores
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
       
        cv2.imshow("yolo11l", frame)
        
        
        if cv2.waitKey(30) == 27:  # 27 bhaneko esc key
            break
    
    print(frame.shape)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()