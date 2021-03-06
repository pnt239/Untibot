#!/usr/bin/env python
"""
Creates an HTTP server with basic auth and websocket communication.
"""
import argparse
import base64
import hashlib
import os
import time
import threading
import webbrowser
import numpy as np
import cv2
import picamera
from picamera.array import PiRGBArray
from PIL import Image
from datetime import datetime
import RPi.GPIO as GPIO
from imutils.object_detection import non_max_suppression
import io as ioEx

try:
    import cStringIO as io
except ImportError:
    import io

import tornado.web
import tornado.websocket
import signal
from tornado.ioloop import PeriodicCallback
from tornado.escape import json_decode
from video import create_capture
from common import clock, draw_str

# Hashed password for comparison and a cookie for login cache
ROOT = os.path.normpath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "password.txt")) as in_file:
    PASSWORD = in_file.read().strip()
COOKIE_NAME = "camp"

uploadDir = os.path.join(ROOT, "upload/")
thread1 = None
detector = None
args = None
resolutions = {"high": (1280, 720), "medium": (640, 480), "low": (320, 240), "360" : (480, 360)}

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.LOW)

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
nested = cv2.CascadeClassifier("haarcascade_eye.xml")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class RecordVideo(threading.Thread):
    """
    Thread checking URLs.
    """

    def __init__(self, camera, camera_width, camera_height):
        """
        Constructor.

        @param urls list of urls to check
        @param output file to write urls output
        """
        threading.Thread.__init__(self)
        self._camera = camera
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._writer_lock = threading.Lock()
        self._frame = None

        self._fourcc = None
        self._writer = None
        self._width = camera_width
        self._height = camera_height
        self._back = np.zeros((320,240,3), dtype="uint8")
        self._is_ready = False
        self._is_recorded = False

    def run(self):
        """
        Thread run method. Check URLs one by one.
        """

        # initialize the video stream and allow the camera
        # sensor to warmup
        print("[Recorder] warming up camera...")
        time.sleep(2.0)

        self._fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        #ret, frame = self._camera.read()
        #(self._height, self._width) = frame.shape[:2]
        print("[Recorder] can start")
        self._is_ready = True
        ret = False
        frame = None
        stream = ioEx.BytesIO()
        
        while (not self._stop.is_set()):
            if args.use_usb:
                ret, frame = self._camera.read()
            else:
                ret = True
                self._camera.capture(stream, format='jpeg', use_video_port=True)
                # Construct a numpy array from the stream
                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                # "Decode" the image from the array, preserving colour
                frame = cv2.imdecode(data, 1)

            if ret==True:
                self._frame_lock.acquire()
                self._frame = frame
                self._frame_lock.release()

                # write the flipped frame
                self._writer_lock.acquire()
                if not (self._writer is None):
                    self._writer.write(frame)
                self._writer_lock.release()
                time.sleep(0.001)

        if not (self._writer is None):
            self._writer.release()
        print('[Recorder] end thread')

    def stop(self):
        self._stop.set()
        print('[Recorder] stop thread')

    def stopped(self):
        return self._stop.isSet()

    def getImage(self):
        clone = None
        if not (self._frame is None):
            self._frame_lock.acquire()
            clone = self._frame.copy()
            self._frame_lock.release()
        else:
            clone = self._back.copy()
        return clone

    def isRecorded(self):
        return self._is_recorded

    def snap(self):
        clone = None
        if not (self._frame is None):
            self._frame_lock.acquire()
            clone = self._frame.copy()
            self._frame_lock.release()
        else:
            clone = self._back.copy()

        fileName =  time.strftime("%Y%m%d-%H%M%S") + '.jpg'
        cv2.imwrite(uploadDir + fileName, clone)
        return fileName

    def startRecord(self):
        if self._is_ready and not self._is_recorded:
            self._writer_lock.acquire()
            self._writer = cv2.VideoWriter(uploadDir + time.strftime("%Y%m%d-%H%M%S") + ".avi", self._fourcc, 20.0, (self._width, self._height), True)
            self._writer_lock.release()
            self._is_recorded = True
            return True
        return False

    def stopRecord(self):
        if self._is_ready and self._is_recorded:
            self._writer_lock.acquire()
            self._writer.release()
            self._writer = None
            self._writer_lock.release()
            self._is_recorded = False
            return True
        return False

class MotionDetection(threading.Thread):
    """
    Thread checking URLs.
    """

    def __init__(self, video):
        """
        Constructor.

        @param urls list of urls to check
        @param output file to write urls output
        """
        threading.Thread.__init__(self)
        self._stop = threading.Event()
        self._pause = True
        self._video = video
        #self._fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        #self._fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        self._fgbg = cv2.createBackgroundSubtractorMOG2()
        self._init = True
        self._hasMotion = False

    def run(self):
        """
        Thread run method. Check URLs one by one.
        """
        pre_stop = False
        begin_t = 0
        end_t = 0

        self.removeNoise()

        while (not self.stopped()):

            if self._pause:
                time.sleep(0.001)
                continue

            frame = self._video.getImage()

            if not (frame is None):
                fgmask = self._fgbg.apply(frame)
                hist = cv2.calcHist([fgmask],[0],None,[256],[0,256])

                white_count = hist[255]

                if (white_count > 500):
                    if not self._video.isRecorded() and not self._pause:
                        if self._video.startRecord():
                            self._hasMotion = True
                            print('[Detector] start record video')
                        else:
                            print('[Detector] start record video fail!')
                    pre_stop = False
                elif (white_count <= 100) and self._video.isRecorded():
                    if not pre_stop:
                        pre_stop = True
                        begin_t = clock()
                    else:
                        end_t = clock()
                        if end_t - begin_t > 10:
                            if self._video.stopRecord():
                                self._hasMotion = False
                                print('[Detector] stop record video')
                            else:
                                print('[Detector] stop record video fail!')
        if self._video.isRecorded():
            self._hasMotion = False
            self._video.stopRecord()
        print('[Detector] end Thread')

    def pause(self):
        self._pause = True
        self._video.stopRecord()
        self._hasMotion = False
        print('[Detector] pause thread')

    def resume(self):
        self._pause = False
        self.removeNoise()
        print('[Detector] resume thread')

    def stop(self):
        self._stop.set()
        print('[Detector] stop thread')

    def stopped(self):
        return self._stop.isSet()

    def removeNoise(self):
        frame = self._video.getImage()
        self._fgbg.apply(frame)
        
    def hasMotion(self):
        return self._hasMotion

class IndexHandler(tornado.web.RequestHandler):

    def get(self):
        #if args.require_login and not self.get_secure_cookie(COOKIE_NAME):
        #    self.redirect("/login")
        #else:
        self.render("index.html")

class FileManagerHandler(tornado.web.RequestHandler):

    def get(self):
        #if args.require_login and not self.get_secure_cookie(COOKIE_NAME):
        #    self.redirect("/login")
        #else:
        self.render("filemanager.html")

class SnapHandler(tornado.web.RequestHandler):
    """docstring for SnapHandeler"""
    def post(self):
        fileName = thread1.snap()
        self.write(fileName)

class RecordHandler(tornado.web.RequestHandler):
    """docstring for SnapHandeler"""
    def post(self):
        json_obj = json_decode(self.request.body)

        if (json_obj['status'] == 'start'):
            thread1.startRecord()
            self.write('Recorder started!')
        else:
            thread1.stopRecord()
            self.write('Recorder stopped!')

class DetectHandler(tornado.web.RequestHandler):
    """docstring for SnapHandeler"""
    def post(self):
        json_obj = json_decode(self.request.body)

        if (json_obj['status'] == 'start'):
            detector.resume()
            self.write('Detector started!')
        else:
            detector.pause()
            self.write('Detector stopped!')

class LoginHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("login.html")

    def post(self):
        password = self.get_argument("password", "")
        if hashlib.sha512(password).hexdigest() == PASSWORD:
            self.set_secure_cookie(COOKIE_NAME, str(time.time()))
            self.redirect("/")
        else:
            time.sleep(1)
            self.redirect(u"/login?error")

class WebSocket(tornado.websocket.WebSocketHandler):

    def on_message(self, message):
        """Evaluates the function pointed to by json-rpc."""

        # Start an infinite loop when this is called
        if message == "read_camera":
            self.camera_loop = PeriodicCallback(self.loop, 10)
            self.camera_loop.start()

        # Extensibility for other methods
        else:
            print("Unsupported function: " + message)

    def loop(self):
        """Sends camera images in an infinite loop."""
        sio = io.StringIO()

        if True:
            img = thread1.getImage()

            t = clock()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #fgmask = fgbg.apply(frame)
            #hist = cv2.calcHist([fgmask],[0],None,[256],[0,256])
            gray = cv2.equalizeHist(gray)

            rects = detect(gray, cascade)

            if len(rects) == 0:
                #print "List is empty"
                # detect people in the image
                (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
                    padding=(8, 8), scale=1.05)
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                
            if detector.hasMotion():
                if len(rects) == 0:
                    GPIO.output(18, GPIO.LOW)
                else:
                    GPIO.output(18, GPIO.HIGH)
                    

            draw_rects(img, rects, (0, 255, 0))
            #if not self.nested.empty():
            #    for x1, y1, x2, y2 in rects:
            #        roi = gray[y1:y2, x1:x2]
            #        vis_roi = vis[y1:y2, x1:x2]
            #        subrects = detect(roi.copy(), self.nested)
            #        draw_rects(vis_roi, subrects, (255, 0, 0))
            dt = clock() - t

            draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))
            #draw_str(fgmask, (20, 20), 'white count: %02d' % hist[255])


            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #img = Image.fromarray(fgmask, mode='L')
            img.save(sio, "JPEG")

        try:
            self.write_message(base64.b64encode(sio.getvalue()))
        except tornado.websocket.WebSocketClosedError:
            self.camera_loop.stop()

def main():
    global thread1
    global detector
    global args
    # Commandline parser
    parser = argparse.ArgumentParser(description="Starts a webserver that "
                        "connects to a webcam.")
    parser.add_argument("--port", type=int, default=8000, help="The "
                        "port on which to serve the website.")
    parser.add_argument("--resolution", type=str, default="low", help="The "
                        "video resolution. Can be high, medium, or low.")
    parser.add_argument("--require-login", action="store_true", help="Require "
                        "a password to log in to webserver.")
    parser.add_argument("--use-usb", action="store_true", help="Use a USB "
                        "webcam instead of the standard Pi camera.")
    args = parser.parse_args()

    camera = None
    camera_width = None
    camera_height = None

    # Select camera
    if args.use_usb:
        camera = cv2.VideoCapture(0)
    else:
        camera = picamera.PiCamera()
        camera.start_preview()

    

    if args.resolution in resolutions:
        camera_width, camera_height = resolutions[args.resolution]

        if args.use_usb:
            camera.set(3, camera_width)
            camera.set(4, camera_height)
        else:
            camera.resolution = resolutions[args.resolution]
    else:
        raise Exception("%s not in resolution options." % args.resolution)

    # Web config
    handlers = [(r"/", IndexHandler),
                (r"/login", LoginHandler),
                (r"/websocket", WebSocket),
                (r"/snap", SnapHandler),
                (r"/record", RecordHandler),
                (r"/detect", DetectHandler),
                (r"/fm", FileManagerHandler)]

    settings = {
        "blog_title": u"Untibot App",
        "cookie_secret":PASSWORD,
        "template_path":os.path.join(os.path.dirname(__file__), "templates"),
        "static_path":os.path.join(os.path.dirname(__file__), "public")
    }

    application = tornado.web.Application(handlers, **settings)
    application.listen(args.port)

    #webbrowser.open("http://localhost:%d/" % args.port, new=2)

    # Create Recorder and Detector threads
    thread1 = RecordVideo(camera, camera_width, camera_height)
    detector = MotionDetection(thread1)

    ioloop = tornado.ioloop.IOLoop.instance()

    try:
        thread1.start()
        detector.start()
        ioloop.start()
        pass
    except KeyboardInterrupt:
        detector.stop()
        thread1.stop()
        ioloop.stop()
        GPIO.cleanup()
        pass
    else:
        pass
    finally:
        pass

if __name__ == "__main__":
    main()
