import cv2

video_name = 'video.mp4'


cap = cv2.VideoCapture(video_name)
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fps = cap.get(cv2.CAP_PROP_FPS)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

print("length: ", length)
print("height", height)
print("width: ", width)
# print(length)