import cv2
import os

# Try opening the physical webcam
cap = cv2.VideoCapture(0)
is_fallback = False

if not cap.isOpened():
    print("Webcam not detected. Attempting to fall back to sample video simulation...")
    fallback_video = "input/sample_video.mp4"
    if os.path.exists(fallback_video):
        cap = cv2.VideoCapture(fallback_video)
        is_fallback = True
        if cap.isOpened():
            print(f"Success: Simulating webcam using {fallback_video}")
        else:
            print(f"Error: Could not open fallback video at {fallback_video}")
            exit()
    else:
        print("Error: Webcam not found and fallback sample_video.mp4 is missing.")
        exit()

print("Camera source opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        if is_fallback:
            # Loop the video if it reaches the end in simulation mode
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            print("Error: could not read frame")
            break

    # Scale down simulation window if it is too large for standard screens
    if is_fallback:
        h, w = frame.shape[:2]
        if w > 800:
            frame = cv2.resize(frame, (800, int(h * 800 / w)))
            
        # Draw overlay text to indicate simulation
        cv2.putText(frame, "SIMULATED WEBCAM (Fallback)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    cv2.imshow("Webcam - Original", frame)
    cv2.imshow("Webcam - Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
