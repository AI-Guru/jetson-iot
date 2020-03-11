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


# Valid values for camera and network.
valid_cameras = ["onboard", "usb"]
valid_networks = ["detect", "segment", "none"]

# Define command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--camera",
    type=str, 
    required=False,
    default = valid_cameras[0],
    help="Defines the camera. Possible values: {}.".format(", ".join(valid_cameras))
)
parser.add_argument(
    "--network",
    type=str, 
    required=False,
    default = valid_networks[0],
    help="Defines the network. Possible values: {}.".format(", ".join(valid_networks))
)

# Parse arguments and handle invalid values.
args = parser.parse_args()
if args.camera not in valid_cameras:
    print("ERROR! Invalid value for camera argument.")
    print("Possible values: {}.".format(", ".join(valid_cameras)))
    exit(0)
if args.network not in valid_networks:
    print("ERROR! Invalid value for network argument.")
    print("Possible values: {}.".format(", ".join(valid_networks)))
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


# Loads the neural network.
print("Loading network...")
if args.network == "detect":
    network_name = "ssd-mobilenet-v2"
    net = jetson.inference.detectNet(network_name, threshold=0.5)
    print("Using network {}.".format(network_name))
elif args.network == "segment":
    network_name = "fcn-resnet18-sun"
    net = jetson.inference.segNet(network_name)
    net.SetOverlayAlpha(100)
    print("Using network {}.".format(network_name))

# Create robot.
robot = Robot()
    
# Light.
light_on = False
gpio.setwarnings(False)
gpio.setmode(gpio.BCM)
gpio.setup(13, gpio.OUT, initial=gpio.LOW)
gpio.setup(21, gpio.OUT, initial=gpio.LOW)


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
        
        # Apply Neural Network.
        if args.network == "detect":
            detections = net.Detect(img, width, height)
        if args.network == "segment":
            net.Process(img, width, height, "")
            net.Overlay(img, width, height, "linear")
        
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
        

# Endpoint for gamepad axes.
@app.route("/set_axis_values", methods=["POST"])
def set_axis_values():

    # Get axis values from form.
    axis_values = [float(x) for x in request.form.getlist("data[]")]
    
    # Extract the axis values that we need.
    left_axis_value = axis_values[1]
    right_axis_value = axis_values[3]
    
    # Do not run motors if activation is too small.
    threshold = 0.2
    if np.abs(left_axis_value) < threshold:
        left_axis_value = 0.0
    if np.abs(right_axis_value) < threshold:
        right_axis_value = 0.0
    
    # Map values to motors.
    scale = -0.5
    robot.left_motor.value = scale * left_axis_value
    robot.right_motor.value = scale * right_axis_value


    # Done.
    return "Success!", 200


# Endpoint for gamepad buttons.
@app.route("/set_button_values", methods=["POST"])
def set_button_values():

    # Get axis values from form.
    button_values = [True if x == "true" else False for x in request.form.getlist("data[]")]

    # Turn on the light.
    if button_values[8] == True:
        return light()
    
    # Done.
    return "Success!", 200


# Endpoint for stopping the robot.
@app.route("/stop", methods=["POST"])
def stop():
    
    robot.left_motor.value = 0.0
    robot.right_motor.value = 0.0
    
    return "Success!", 200


# Endpoint for toggling the light.
@app.route("/light", methods=["POST"])
def light():
    
    global light_on
    light_on = not light_on
    
    if light_on == True:
        gpio.output(13, gpio.HIGH)
        gpio.output(21, gpio.HIGH)
    else:
        gpio.output(13, gpio.LOW)
        gpio.output(21, gpio.LOW)
    
    return "Success!", 200


# Endpoint for status.
@app.route("/status")
def status():
    status = {
        "left motor": robot.left_motor.value,
        "right motor": robot.right_motor.value
    }

    return jsonify(status), 200
  
    
# Main.
if __name__ == '__main__':
    main()
