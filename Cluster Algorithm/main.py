import Open_Img
import K_Means
import cv2
import datetime


if __name__ == "__main__":
    img = Open_Img.Open_JPG_OpenCV(r"D:\Python_Learn\Cluster_Algorithm\001.jpg")
    start_time = datetime.datetime.now()

    # dst = K_Means.K_Means_OpenCV(5, img)
    dst = K_Means.K_Means_sklearn(3, img)
    # dst = K_Means.K_Means_sklearn_Auto(2, 10, img)

    end_time = datetime.datetime.now()
    print(end_time - start_time)

    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()