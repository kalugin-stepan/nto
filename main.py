import pymurapi as mur
import cv2
import time
import numpy as np
from pid import PID, clamp
from config import *

STABILIZING_H = 0
STABILIZING_ON_MARKER = 1
CHOOSING_DIRECTION = 2
STABILIZING_ON_DIRECTION = 3
GO = 4

auv = mur.mur_init()

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
arucoParams = cv2.aruco.DetectorParameters_create()

state = STABILIZING_H

t0 = time.time()

pidv = PID(Pv, Iv, Dv)
pida = PID(Pa, Ia, Da)
pidh = PID(Ph, Ih, Dh)

global h0
h0 = 0

def h_is_stabilized(h: float, h0: float, dt: float) -> bool:
    hv = (h - h0)/dt
    return abs(h - H0) <= OK_EH and abs(hv) <= OK_VH

def hold_h(dt: float) -> bool:
    global h0
    h = auv.get_depth()
    eh = h - H0
    v = pidh.process(eh, dt)
    auv.set_motor_power(2, v)
    auv.set_motor_power(3, v)
    return h_is_stabilized(h, h0, dt)

global a0, y0

a0 = 0
y0 = 0

def stabilize_on_point(ex: float, ey: float, dt: float) -> bool:
    global a0, y0

    ea = np.rad2deg(np.arctan(ex/ey))
    if ea > 0 and ex < 0:
        ea -= 180
    if ea < 0 and  ex > 0:
        ea += 180

    if abs(ea) > OK_EA and abs(ey) > OK_EY:
        av = pida.process(ex, dt)
        print(ea, av)
        auv.set_motor_power(0, -av)
        auv.set_motor_power(1, av)
    if abs(ea) <= OK_EA:
        v = pidv.process(ey, dt)
        auv.set_motor_power(0, v)
        auv.set_motor_power(1, v)
    
    vy = (ey - y0)/dt
    av = (ea - a0)/dt

    is_stabilized = abs(vy) <= OK_VY and abs(av) <= OK_AV and abs(ey) <= OK_EY

    y0 = ey
    a0 = ea
    return is_stabilized

def stabilize_on_angle(ex: float, dt: float) -> bool:
    global a0
    av = pida.process(ex, dt)
    auv.set_motor_power(0, -av)
    auv.set_motor_power(1, av)
    av = (ex - a0)/dt

    is_stabilized = abs(av) <= OK_AV and abs(ex) <= OK_EA

    a0 = ex

    return is_stabilized

def stabilize_on_marker(img, dt: float) -> bool:
    img_w = img.shape[1]
    img_h = img.shape[0]
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)

    if len(corners) != 1:
        return False
    x = 0
    y = 0
    for i in corners[0][0]:
        x += i[0]
        y += i[1]
    x /= 4
    y /= 4

    ex = img_w/2 - x
    ey = img_h/2 - y

    return stabilize_on_point(ex, ey, dt)

def choose_direction(img):
    return 'cock'
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cnts, _ = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    img_cnts = cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)

    cv2.imshow('cnts', img_cnts)

    x = 0
    y = 0

    j = 0

    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        x0 = approx[0][0][0]
        y0 = approx[0][0][1]
        x1 = approx[1][0][0]
        y1 = approx[1][0][1]
        x2 = approx[3][0][0]
        y2 = approx[3][0][1]
        dx01 = x0 - x1
        dy01 = y0 - y1
        dx02 = x0 - x2
        dy02 = y0 - y2
        a = dx01*dx01 + dy01*dy01
        b = dx02*dx02 + dy02*dy02
        if len(approx) == 4 and max(a, b)/min(a, b) > 9:
            for j in range(4):
                x += approx[j][0][0]
                y += approx[j][0][1]
            x //= 4
            y //= 4
            color = img_hsv[x, y]

            img_hsv = cv2.drawContours(img_hsv, cnts, j, (0, 255, 0), 3)

            return np.array([max(color[0]-5, 0), 0, 0], dtype=np.uint8), np.array([min(color[0]+5, 180), 255, 255], dtype=np.uint8)
        j += 1
    return None

def stabilize_on_direction(img, color_range) -> bool:
    img_w = img.shape[1]
    img_h = img.shape[0]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_color_range_mask = cv2.inRange(img_hsv, color_range[0], color_range[1])

    cv2.imshow('img_color_range_mask', img_color_range_mask)
    cv2.waitKey(10)

    cnts, _ = cv2.findContours(img_color_range_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(cnts) < 1: return False

    approx = cv2.approxPolyDP(cnts[0], 0.02 * cv2.arcLength(cnts[0], True), True)

    if len(approx) != 4: return False

    x = 0
    y = 0

    for i in range(4):
        x += approx[i][0][0]
        y += approx[i][0][1]
    
    x //= 4
    y //= 4

    ex = img_w/2 - x
    ey = img_h/2 - y

    return stabilize_on_angle(ex, dt)

cur_color_range = None

# i = 0

while True:
    time.sleep(0.03)
    img = auv.get_image_bottom()

    t = time.time()
    dt = t - t0
    t0 = t

    h = auv.get_depth()

    if state != STABILIZING_H and not h_is_stabilized(h, h0, dt):
        state = STABILIZING_H

    if state == STABILIZING_H:
        hold_h(dt)
        if h_is_stabilized(h, h0, dt):
            auv.set_motor_power(2, 0)
            auv.set_motor_power(3, 0)
            state = STABILIZING_ON_MARKER
    elif state == STABILIZING_ON_MARKER:
        if stabilize_on_marker(img, dt):
            auv.set_motor_power(0, 0)
            auv.set_motor_power(1, 0)
            state = CHOOSING_DIRECTION
    elif state == CHOOSING_DIRECTION:
        cur_color_range = choose_direction(img)
        if cur_color_range != None:
            cur_color_range = (np.array([50, 0, 0], dtype=np.uint8), np.array([70, 255, 255], dtype=np.uint8))
            state = STABILIZING_ON_DIRECTION
    elif state == STABILIZING_ON_DIRECTION:
        if stabilize_on_direction(img, cur_color_range):
            state = GO
            auv.set_motor_power(0, 0)
            auv.set_motor_power(1, 0)
    elif state == GO:
        print('go')
    h0 = h

    cv2.putText(img, str(state), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # img_w = img.shape[1]
    # img_h = img.shape[0]
    # corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    # img = cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # if len(corners) != 1:
    #     continue
    # x = 0
    # y = 0
    # for j in corners[0][0]:
    #     x += j[0]
    #     y += j[1]
    # x /= 4
    # y /= 4

    # if i == 10:
    #     ex = img_w/2 - x
    #     ey = img_h/2 - y

    #     print(ex, ey)
    #     a = np.rad2deg(np.arctan(ex/ey))
    #     if a > 0 and ex < 0:
    #         a -= 180
    #     if a < 0 and  ex > 0:
    #         a += 180
    #     print(a)
    #     i = 0

    cv2.imshow('', img)
    cv2.waitKey(10)

    # i += 1