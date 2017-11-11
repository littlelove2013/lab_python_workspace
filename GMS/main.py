import cv2
import GMS

def main():
    # 获得视频的格式
    videoCapture = cv2.VideoCapture(0)
    videoCapture2 = cv2.VideoCapture(1)
    # 获得码率及尺寸
    # fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
    # size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
    #         int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    # # 指定写视频的格式, I420-avi, MJPG-mp4
    # videoWriter = cv2.VideoWriter('oto_other.mp4', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
    # 读帧
    while 1:
        success, frame1 = videoCapture.read()  # 获取下一帧
        success2, frame2 = videoCapture2.read()  # 获取下一帧
        gms=GMS.GMS(frame1,frame2)
        #gmsmatchimg = gms.getGmsMatchesImg()
        cv2.imshow('src Video', frame1)  # 显示
        cv2.imshow('src2 Video', frame2)  # 显示
        #cv2.imshow('gms Video', gmsmatchimg)  # 显示
        cv2.waitKey(10)  # 延迟
        # videoWriter.write(frame)  # 写视频帧
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()