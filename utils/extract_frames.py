import cv2
import os
import sys


def main():

    video_reader = cv2.VideoCapture(sys.argv[1])
    base_path = sys.argv[2]
    os.makedirs(base_path, exist_ok=True)

    cnt = 0

    print('Proceeding:', sys.argv[1])
    while video_reader.isOpened():
        flag, img = video_reader.read()
        if not flag:
            break
        cv2.imwrite(os.path.join(base_path, '{:06d}.jpg'.format(cnt)), img)

        cnt += 1

    video_reader.release()

if __name__ == '__main__':
    main()
