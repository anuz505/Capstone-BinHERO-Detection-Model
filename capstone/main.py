import cv2
from ultralytics import YOLO
import supervision as sv
import time
import torch
import sys
import os

from utils import parse_arguments

def main():
    args = parse_arguments()
    frame_width , frame_height = args.webcam_res 
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            device = "cpu"
            print("No GPU detected, using CPU")
    else:
        device = args.device
        print(f"using device: {device}")

    if not args.video and not args.webcam:
        print("Error: Please specify either --video or --webcam")
        sys.exit(1)
    
    if args.video and args.webcam:
        print("Error: Please specify either --video or --webcam, not both")
        sys.exit(1)
    

    if args.webcam:
        cap = cv2.VideoCapture(0)
        print("Using the webcam")
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    elif args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Using video file: {args.video}")

 # Load model and move to GPU
    model = YOLO("./models/best.pt")
    model.to(device)  # Move model to GPU
    print(f"Model loaded on: {device}")

    # annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1,
        text_color=sv.Color.BLACK,
        text_position=sv.Position.TOP_LEFT
    )
    confidence_threshold = 0.6
    max_box_area_ratio  = 0.6 
    pTime = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("error: couldn't read the frame")
            break
        result = model(frame,agnostic_nms =  True)[0]
        detections = sv.Detections.from_ultralytics(result)

        # filtering with confidence thresholds 
        detections = detections[detections.confidence > confidence_threshold]

        if len(detections) > 0:
            frame_area = frame.shape[0] * frame.shape[1]
            box_areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            area_ratios = box_areas / frame_area
            size_mask  = area_ratios < max_box_area_ratio 
            detections = detections[size_mask]
        
        # labels with class names and their confidence scores
        labels = [
             f"{model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        frame = box_annotator.annotate(scene=frame,detections=detections)
        frame = label_annotator.annotate(scene=frame,detections=detections, labels=labels)

        ctime = time.time()
        fps = 1 / (ctime - pTime)
        pTime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
        
        cv2.imshow("yolo11",frame)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()