import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened(), flush=True)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame", flush=True)
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
