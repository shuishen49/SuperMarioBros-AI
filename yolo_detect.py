import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Image path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='YOLO weights path')
    parser.add_argument('--conf', type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.source)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')

    model = YOLO(args.weights)
    results = model.predict(source=str(image_path), conf=args.conf, verbose=False)

    plotted = results[0].plot()
    out_path = image_path.with_name(image_path.stem + '_detected.png')
    cv2.imwrite(str(out_path), plotted)

    print(f'Detection result saved to: {out_path}')
    if results[0].boxes is not None:
        print(results[0].boxes)


if __name__ == '__main__':
    main()
