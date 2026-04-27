import cv2


cap = cv2.VideoCapture(0)
cap.set(6, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:

    ret, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(fps)

    cv2.imshow('USB Camera', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()
