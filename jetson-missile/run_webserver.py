import jetson.inference
import jetson.utils
from jetbot import Robot
from flask import Response
from flask import Flask
from flask import render_template
from flask import request
from flask import make_response
from flask import jsonify
import threading
import time
import cv2
import ctypes
import argparse
import numpy as np
import RPi.GPIO as gpio
import missilecontrol


valid_cameras = ["onboard", "usb"]

# Define command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--camera",
    type=str, 
    required=False,
    default = valid_cameras[0],
    help="Defines the camera. Possible values: {}.".format(", ".join(valid_cameras))
)

# Parse arguments and handle invalid values.
args = parser.parse_args()
if args.camera not in valid_cameras:
    print("ERROR! Invalid value for camera argument.")
    print("Possible values: {}.".format(", ".join(valid_cameras)))
    exit(0)

# A lock for multithreading.
lock = threading.Lock()

# The outputframe that will be available via the stream.
output_frame = None

# Creates the flask-app.
app = Flask(__name__)

# Creates the camera.
print("Creating camera...")
width, height = 1280, 720
if args.camera == "onboard":
    camera = jetson.utils.gstCamera(width, height, "0")
    print("Using onboard camera.")
elif args.camera == "usb":
    camera = jetson.utils.gstCamera(width, height, "/dev/video1")
    print("Using USB camera.")
img_output = jetson.utils.cudaAllocMapped(width * height * 4 * ctypes.sizeof(ctypes.c_float))

# Creates missile control.
missile_control = missilecontrol.MissileControl()

# The main method.
def main():
    ip = "0.0.0.0"
    port = "8666"
    
    # Starts the processing thread.
    print("Start processing...")
    start_processing()

    # Start the flask app.
    print("Starting flask-app...")
    app.run(
        host=ip,
        port=port,
        debug=True,
        threaded=True, 
        use_reloader=False)
    
    
# Starts the processing thread.
def start_processing():
    t = threading.Thread(target=process_frame)
    t.daemon = True
    t.start()

    
# Process a frame.
def process_frame():
    global output_frame, lock
    
    while True:
        
        # Grab a frame from the camera.
        img, width, height = camera.CaptureRGBA(zeroCopy=1)
        
        # Synchronize CUDA device.
        jetson.utils.cudaDeviceSynchronize()
        
        # Lock and update image.
        with lock:
            img = jetson.utils.cudaToNumpy(img, width, height, 4)
            final_output = img
            final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
            output_frame = final_output.copy()
            
        time.sleep(0.01)
        

# Shows the main page.
@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


# Endpoint for the video stream.
@app.route("/video_image")
def video_image():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(
        get_video_image(),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )

# Yields the video image.
def get_video_image():
    global output_frame, lock

    # Infinite loop for yielding images.
    while True:
        
        # Wait for the lock.
        with lock:
            
            # Skip if there is no frame.
            if output_frame is None:
                continue

            # Encode the frame in JPEG format.
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            # Ensure the frame was successfully encoded
            if not flag:
                continue

        # Yield a byte stream.
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        


@app.route("/left", methods=["POST"])
def left():

    missile_control.turret_left()
    time.sleep(1.0)
    missile_control.turret_stop()

    # Done.
    return "Success!", 200

@app.route("/right", methods=["POST"])
def right():

    missile_control.turret_right()
    time.sleep(1.0)
    missile_control.turret_stop()
    
    # Done.
    return "Success!", 200

@app.route("/fire", methods=["POST"])
def fire():

    missile_control.turret_fire()

    # Done.
    return "Success!", 200


# Endpoint for status.
@app.route("/status")
def status():
    status = {
    }

    return jsonify(status), 200
  
    
# Main.
if __name__ == '__main__':
    main()
