import cv2
import numpy as np

RECTANGLE = 1
TRIANGLE = 2
CIRCLE = 3

# img = cv2.imread('img_rotated.png')
img = cv2.imread('img.png')

img_w = img.shape[1]
img_h = img.shape[0]

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_hsv_black = (0, 0, 0)
high_hsv_black = (180, 255, 20)

black_mask_hsv = cv2.inRange(img_hsv, low_hsv_black, high_hsv_black)

all_cnts, ir = cv2.findContours(black_mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts = []

ir = ir[0]

i = 0

for el in ir:
    if el[3] == -1 or ir[el[3]][3] == -1:
        i += 1
        continue
    cnts.append(all_cnts[i])
    i += 1

i = 0

k = None
b = None

for c in cnts:
    approx = cv2.approxPolyDP(c, cv2.arcLength(c, True) * 0.02, True)
    if len(approx) == 4:
        x0 = approx[0][0][0]
        y0 = approx[0][0][1]
        x1 = approx[1][0][0]
        y1 = approx[1][0][1]
        x2 = approx[3][0][0]
        y2 = approx[3][0][1]

        dx1 = x0 - x1
        dy1 = y0 - y1
        dx2 = x0 - x2
        dy2 = y0 - y2

        a = np.sqrt(dx1*dx1 + dy1*dy1)
        b = np.sqrt(dx2*dx2 + dy2*dy2)

        if a < b:
            a, b = b, a
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            dx1, dx2 = dx2, dx1
            dy1, dy2 = dy2, dy1

        if 5 < a/b:
            cv2.drawContours(img, cnts, i, (0, 255, 0), 3)
            k = -dx1/dy1
            cx = (x0 + x1)/2
            cy = (y0 + y1)/2
            b = cy - k*cx
            for x in np.arange(0, img_w, 0.01):
                y = k*x + b
                img = cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 3)
            break
    i += 1

j = 0
first_object = None
first_object_index = None
first_object_x = None
first_object_y = None
first_object_min_r = img_w*img_w + img_h*img_h
kp = -1/k
bp = None

for c in cnts:
    if j == i:
        j += 1
        continue
    
    ((rx, ry), (rw, rh), ra) = cv2.minAreaRect(c)

    r = abs(k*rx + b - ry)

    if r < first_object_min_r:
        first_object_min_r = r
        first_object = c
        first_object_index = j
        first_object_x = rx
        first_object_y = ry
        bp = ry - kp*rx

    j += 1

for x in np.arange(0, img_w, 0.01):
    y = kp*x + bp
    img = cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 3)

img = cv2.circle(img, (int(first_object_x), int(first_object_y)), 3, (0, 0, 255), 3)

approx = cv2.approxPolyDP(first_object, cv2.arcLength(first_object, True) * 0.02, True)

object_type = None

if len(approx) == 3:
    object_type = TRIANGLE
elif len(approx) == 4:
    object_type = RECTANGLE
else:
    object_type = CIRCLE

objects = [(first_object, object_type)]

second_object = None
second_object_x = None
second_object_y = None
second_object_min_r = img_w*img_w + img_h*img_h
second_object_index = None
third_object = None
third_object_x = None
third_object_y = None
third_object_min_r = img_w*img_w + img_h*img_h
third_object_index = None

i = 0

for c in cnts:
    if i == first_object_index:
        i += 1
        continue
    ((rx, ry), (rw, rh), ra) = cv2.minAreaRect(c)
    r = abs(kp*rx + b - ry)
    if r < second_object_min_r:
        if second_object is not None and (third_object is None or third_object_min_r > second_object_min_r):
            third_object_min_r = second_object_min_r
            third_object_x = second_object_x
            third_object_y = second_object_y
            third_object = second_object
        second_object_min_r = r
        second_object = c
        second_object_x = rx
        second_object_y = ry
        second_object_index = i
    elif r < third_object_min_r:
        third_object = c
        third_object_x = rx
        third_object_y = ry
        third_object_min_r = r
        third_object_index = i
    i += 1

img = cv2.drawContours(img, cnts, second_object_index, (0, 255, 0), 3)
img = cv2.drawContours(img, cnts, third_object_index, (0, 255, 0), 3)

while True:
    cv2.imshow('', img)
    cv2.waitKey(10)