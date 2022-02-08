import Open_Img
import K_Means
import cv2


if __name__ == "__main__":
    img = Open_Img.Open_JPG_OpenCV(r"D:\Python_Learn\Cluster_Algorithm\001.jpg")
    # dst = K_Means.K_Means_OpenCV(3, img)
    # dst = K_Means.K_Means_sklearn(3, img)
    dst, k = K_Means.K_Means_sklearn_Auto(2, 10, img)
    print("k取",k,"时，k-means可以取得最好的聚类效果")
    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()