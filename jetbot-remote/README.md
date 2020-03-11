# JetBot Remote.

Allows you to remotely control your JetBoot while running Neural Networks inference.

## Installation and Set-Up.

1. Acquire and assemble a JetBot.
2. Make sure that the following dependencies are installed:
  - [https://github.com/NVIDIA-AI-IOT/jetbot](https://github.com/NVIDIA-AI-IOT/jetbot) (Usually comes preinstalled)
  - [https://github.com/dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
3. Start the web server as follows:
```
python3 run_webserver.py
```
4. When the server is running go to http://<IP-OF-JETBOT>:8666.
5. Connect your gamepad.
6. Enjoy!

## Known issues.
    
### Camera might be upside down.
    
This is an issue of jetson-inference. Please check the project's repository for a solution.
    
### Script does not work because of some camera issues.
    
Restart the camera as follows:
``` 
sh restart_camera.sh
```
