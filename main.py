import cv2 as cv
from ultralytics import YOLO
from utils import draw_rounded_rect, draw_text_with_bg

# Load the YOLO model
try:
    model = YOLO("yolov8x.pt")
    class_names = list(model.names.values())
except Exception as e:
    raise Exception(f"Error loading YOLO model: {e}")

# Target classes and colors
target_classes = {'backpack': (0, 255, 0), 'handbag': (0, 0, 255), 'suitcase': (255, 0, 0)}
class_count = {cls: 0 for cls in target_classes}
counted_ids = {cls: set() for cls in target_classes}

# Video paths
video_path = "DATA/INPUTS/people_in_metro.mp4"
output_path = "DATA/OUTPUTS/tracking_luggage_with_YOLOv8x.mp4"

# Initialize video capture and writer
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error: Couldn't open the video at {video_path}")

# Get video properties safely
try:
    w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
except Exception as e:
    cap.release()
    raise Exception(f"Error setting up video writer: {e}")


# Function to update counts
def update_count(class_name, track_id):
    if track_id not in counted_ids[class_name]:
        class_count[class_name] += 1
        counted_ids[class_name].add(track_id)


# Function to draw bounding boxes and text
def draw_detections(frame, bboxes, track_ids, class_ids, scores):
    for bbox, track_id, class_id, score in zip(bboxes, track_ids, class_ids, scores):
        class_name = class_names[class_id]
        if class_name in target_classes and score > 0.6:
            x1, y1, x2, y2 = map(int, bbox)
            color = target_classes[class_name]
            draw_rounded_rect(frame, (x1, y1, x2, y2), ellipse_color=color, line_thickness=2, ellipse_thickness=6,
                              radius=20)
            draw_text_with_bg(frame, f"{class_name} ID{track_id}", (x1, y1 - 10), font_scale=1, thickness=2,
                              bg_color=color, text_color=(255, 255, 255))
            update_count(class_name, track_id)


# Function to display counts on frame
def display_counts(frame):
    cv.rectangle(frame, (0, 0), (400, 220), (255, 255, 255), cv.FILLED)
    for i, (cls, color) in enumerate(target_classes.items()):
        cv.putText(frame, f"{cls.upper()}: {class_count[cls]}", (10, 50 + 50 * i), cv.FONT_HERSHEY_COMPLEX, 1.5, color,
                   2)


# Main processing loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame read failed or video ended.")
            break

        # YOLO tracking and detection
        try:
            results = model.track(frame, persist=True)
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            continue  # Skip this frame on error

        # Ensure the results contain necessary attributes before proceeding
        if results[0].boxes.id is not None:
            bboxes = results[0].boxes.xyxy.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            scores = results[0].boxes.conf.cpu().tolist()

            if bboxes and track_ids and class_ids and scores:
                draw_detections(frame, bboxes, track_ids, class_ids, scores)
                display_counts(frame)

        # Write output and display resized frame
        out.write(frame)
        resized_frame = cv.resize(frame, (int(0.55 * frame.shape[1]), int(0.55 * frame.shape[0])))
        cv.imshow("Tracking Luggage", resized_frame)

        if cv.waitKey(1) & 0xFF == ord('p'):  # 'p' to pause or exit
            break
except Exception as e:
    print(f"Error during video processing: {e}")
finally:
    # Release resources
    cap.release()
    out.release()
    cv.destroyAllWindows()
    print("Released video resources and closed windows.")
