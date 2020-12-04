# this is the interface side of this hand writing recognizing project
# importing libraries
import numpy
import cv2
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from keras.models import load_model

# setting up video capture
width = 640
height = 480
cameraNo = 0

cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)

# loading pretrained model
model = load_model('handModel.h5')

# reading image
while True:
    success, imgOriginal = cap.read()
    
    # converting image to grayscale
    imgGray = rgb2gray(imgOriginal)
    imgGrayU8 = img_as_ubyte(imgGray)
    
    # thresholding
    (thresh, imgBinary) = cv2.threshold(imgGrayU8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # resizing image
    imgResized = cv2.resize(imgBinary, (28, 28))
    
    # inverting image colors
    imgGrayInvert = 255 - imgResized
    cv2.imshow('invert image', imgGrayInvert)
    
    # reshaping the image for final transmission
    imgFinal = imgGrayInvert.reshape(1, 28, 28, 1)
    
    # transmitting image to model
    ans = model.predict(imgFinal)
    
    # extracting the result from the array returned and printing the pridicted values
    ans = numpy.argmax(ans, axis = 1)[0]
    print(ans)

    # putting the predicted value as a text on webcam feed
    cv2.putText(imgOriginal, 'Predicted Digit : '+str(ans), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
    cv2.imshow('Original Image', imgOriginal)
    
    # handeling exit
    if (cv2.waitKey(1) and 0xFF == ord('q')):
        break

# releasing memory and closing
cap.release()
cv2.destroyAllWindows()