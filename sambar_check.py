import cv2

def check_sambar_consistency(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Couldn't read the image."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur < 100:
        return "🫗 Watery Disaster: Add more dal!"
    elif blur < 300:
        return "🥣 Hotel Style: Perfectly balanced, as all things should be."
    else:
        return "🧱 Cement-Level Thiccness: Dosa might bounce off it!"
