import cv2
from ultralytics import YOLO

def start_crowd_monitor(video_source="people-detection.mp4"):
    print("Loading YOLOv8 AI Vision Model...")
    model = YOLO('yolov8n.pt') 
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return 0

    print("Starting Live Bus Stop Feed. Press 'q' to quit.")

    # --- NEW: Variable to track the peak crowd size ---
    max_people_count = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nVideo stream ended.")
            break

        # Run inference, looking only for people (class 0)
        results = model.predict(frame, classes=[0], conf=0.4, verbose=False)
        
        # Count people in the current frame
        current_count = len(results[0].boxes)

        # --- NEW: Update the max count if the current frame is higher ---
        if current_count > max_people_count:
            max_people_count = current_count

        # Draw boxes and UI on the frame
        annotated_frame = results[0].plot()
        cv2.rectangle(annotated_frame, (10, 10), (550, 80), (0, 0, 0), -1)
        
        # Display both the live count AND the max count on the screen
        cv2.putText(annotated_frame, f"Live: {current_count} | MAX: {max_people_count}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("BMTC Smart Stop Monitor", annotated_frame)

        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- NEW: Print and return the final answer ---
    print("=========================================")
    print(f"📊 FINAL RESULT: Maximum crowd detected was {max_people_count} people.")
    print("=========================================")
    
    # Returning the value so your ML model can use it!
    return max_people_count

if __name__ == "__main__":
    # Ensure you have the video file in your backend folder!
    final_count = start_crowd_monitor("busstop.mp4")