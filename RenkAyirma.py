import cv2
import pyautogui
import numpy as np

width = 1920
pyautogui.FAILSAFE = False

# Renklerin HSV değerlerini belirle
colors = {
    'mavi': ([110, 50, 50], [130, 255, 255]),
    'sari': ([10, 100, 100], [40, 255, 255]),
    'mor': ([70, 100, 100], [169, 255, 255]),
    'kirmizi': ([0, 100, 100], [10, 255, 255]), 
    'yesil': ([38, 100, 100], [75, 255, 255]),
    'beyaz': ([0, 0, 200], [180, 30, 255]),
     
}


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Renk maskeleme işlemi
    masks = {}
    for color, (lower, upper) in colors.items():
        masks[color] = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = sum(masks.values())  # Tüm maskeleri topla
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if len(contour) > 0:  # Konturun noktalarını kontrol et
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Maskeler içindeki renk alanlarını hesapla
            color_areas = {color: np.sum(masks[color][y:y+h, x:x+w] > 0) for color in colors}
            max_color = max(color_areas, key=color_areas.get)
            max_area = color_areas[max_color]

            if max_area > 5000:
                cv2.putText(frame, f"Detected Color: {max_color.capitalize()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                x_center = int(x + w / 2)
                y_center = int(y + h / 2)
                pyautogui.moveTo(x_center, y_center)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
