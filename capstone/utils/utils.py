import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 live")
    parser.add_argument("--webcam-res",default=[1280,720],nargs=2,type=int)
    parser.add_argument("--webcam",action="store_true",help="use webcam")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--device", default="auto", help="Device to run inference on")
    args = parser.parse_args()
    return args