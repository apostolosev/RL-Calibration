import cv2
import glob

frameSize = (512, 512)
out = cv2.VideoWriter('generated.avi',cv2.VideoWriter_fourcc(*'DIVX'), 13.5, frameSize)

frames = glob.glob("data/generated/strean_1690205099698/*.jpg")
frames.sort(key = lambda path: int(path.split("/")[-1].split(".")[0].split("_")[1]))

for frame in frames:
    img = cv2.imread(frame)
    out.write(img)

out.release()