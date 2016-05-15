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

try:
    import cStringIO as io
except ImportError:
    import io

import tornado.web
import tornado.websocket
import signal
from tornado.ioloop import PeriodicCallback
from video import create_capture
from common import clock, draw_str

# Hashed password for comparison and a cookie for login cache
ROOT = os.path.normpath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "password.txt")) as in_file:
    PASSWORD = in_file.read().strip()
COOKIE_NAME = "camp"

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

class VideoRecorder(threading.Thread):
    """docstring for VideoRecorder"""
    def __init__(self):
        threading.Thread.__init__(self)
        self._stop = threading.Event()
        self._fourcc = cv2.VideoWriter_fourcc(*'X264')
        self._out = cv2.VideoWriter('output.mp4', self._fourcc, 20.0, (640,480))
        self._is_record = True

    def run(self):
        while (not self.stopped()) and (camera.isOpened()) and (self._is_record):
            _, frame = camera.read()
            frame = cv2.flip(frame,0)
            # write the flipped frame
            self._out.write(frame)

        self._out.release()

    def stop(self):
        self._stop.set()
        self._is_record = False

    def stopped(self):
        return self._stop.isSet()

class MotionDetection(threading.Thread):
    """
    Thread checking URLs.
    """

    def __init__(self):
        """
        Constructor.

        @param urls list of urls to check
        @param output file to write urls output
        """
        threading.Thread.__init__(self)
        self._stop = threading.Event()
        self.white_count = 0

    def run(self):
        """
        Thread run method. Check URLs one by one.
        """
        while (not self.stopped()):
            _, frame = camera.read()
            fgmask = fgbg.apply(frame)
            hist = cv2.calcHist([fgmask],[0],None,[256],[0,256])
            self.white_count = hist[255]

            if self.white_count > 100:
                #

            time.sleep(300)
            pass

        print('End Thread')

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

class IndexHandler(tornado.web.RequestHandler):

    def get(self):
        if args.require_login and not self.get_secure_cookie(COOKIE_NAME):
            self.redirect("/login")
        else:
            self.render("index.html", port=args.port)


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

        if args.use_usb:
            _, frame = camera.read()
            img = frame#Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            t = clock()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            fgmask = fgbg.apply(frame)
            hist = cv2.calcHist([fgmask],[0],None,[256],[0,256])
            #gray = cv2.equalizeHist(gray)

            #rects = detect(gray, cascade)
            vis = img.copy()
            #draw_rects(vis, rects, (0, 255, 0))
            #if not self.nested.empty():
            #    for x1, y1, x2, y2 in rects:
            #        roi = gray[y1:y2, x1:x2]
            #        vis_roi = vis[y1:y2, x1:x2]
            #        subrects = detect(roi.copy(), self.nested)
            #        draw_rects(vis_roi, subrects, (255, 0, 0))
            dt = clock() - t

            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            draw_str(fgmask, (20, 20), 'white count: %02d' % hist[255])


            #img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            img = Image.fromarray(fgmask, mode='L')
            img.save(sio, "JPEG")
        else:
            camera.capture(sio, "jpeg", use_video_port=True)

        try:
            self.write_message(base64.b64encode(sio.getvalue()))
        except tornado.websocket.WebSocketClosedError:
            self.camera_loop.stop()


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

if args.use_usb:
    import cv2
    from PIL import Image
    camera = cv2.VideoCapture(0)
else:
    import picamera
    camera = picamera.PiCamera()
    camera.start_preview()

#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2()
# Create new threads
thread1 = VideoRecorder()

# Start new Threads
#thread1.start()
#thread1.join()

resolutions = {"high": (1280, 720), "medium": (640, 480), "low": (320, 240), "360" : (480, 360)}
if args.resolution in resolutions:
    if args.use_usb:
        w, h = resolutions[args.resolution]
        camera.set(3, w)
        camera.set(4, h)
    else:
        camera.resolution = resolutions[args.resolution]
else:
    raise Exception("%s not in resolution options." % args.resolution)

handlers = [(r"/", IndexHandler), (r"/login", LoginHandler),
            (r"/websocket", WebSocket),
            (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': ROOT})]
application = tornado.web.Application(handlers, cookie_secret=PASSWORD)
application.listen(args.port)

webbrowser.open("http://localhost:%d/" % args.port, new=2)

ioloop = tornado.ioloop.IOLoop.instance()
#signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(on_shutdown))
try:
    ioloop.start()
    thread1.start()
    thread1.join()
    pass
except KeyboardInterrupt:
    thread1.stop()
    ioloop.stop()
    pass
else:
    pass
finally:
    pass
