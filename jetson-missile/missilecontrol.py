import os
os.environ['PYUSB_DEBUG'] = 'debug'
import sys
import time
import usb.core
import hid


class MissileControl():
   
    def __init__(self, use_hid=True):
        self.use_hid = use_hid
        if self.use_hid == True:
            self.hidraw = hid.device(0x2123, 0x1010)
            self.hidraw.open(0x2123, 0x1010)
        else:
            self.dev = usb.core.find(idVendor=0x2123, idProduct=0x1010)
            if self.dev is None:
                raise ValueError('Launcher not found.')
            if self.dev.is_kernel_driver_active(0) is True:
                self.dev.detach_kernel_driver(0)
            self.dev.set_configuration()


    def turret_up(self):
        if self.use_hid == True:
            self.hidraw.send_feature_report([0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00])
        else:
            self.dev.ctrl_transfer(0x21,0x09,0,0,[0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00])

    def turret_down(self):
        if self.use_hid == True:
            self.hidraw.send_feature_report([0x02,0x04,0x00,0x00,0x00,0x00,0x00,0x00])
        else:
            self.dev.ctrl_transfer(0x21,0x09,0,0,[0x02,0x01,0x00,0x00,0x00,0x00,0x00,0x00])

    def turret_left(self):
        if self.use_hid == True:
            self.hidraw.send_feature_report([0x02,0x04,0x00,0x00,0x00,0x00,0x00,0x00])
        else:
            self.dev.ctrl_transfer(0x21,0x09,0,0,[0x02,0x04,0x00,0x00,0x00,0x00,0x00,0x00])

    def turret_right(self):
        if self.use_hid == True:
            self.hidraw.send_feature_report([0x02,0x08,0x00,0x00,0x00,0x00,0x00,0x00])
        else:
            self.dev.ctrl_transfer(0x21,0x09,0,0,[0x02,0x08,0x00,0x00,0x00,0x00,0x00,0x00])
    
    def turret_stop(self):
        if self.use_hid == True:
            self.hidraw.send_feature_report([0x02,0x20,0x00,0x00,0x00,0x00,0x00,0x00])
        else:
            self.dev.ctrl_transfer(0x21,0x09,0,0,[0x02,0x20,0x00,0x00,0x00,0x00,0x00,0x00])

    def turret_fire(self):
        if self.use_hid == True:
            self.hidraw.send_feature_report([0x02,0x10,0x00,0x00,0x00,0x00,0x00,0x00])
        else:
            self.dev.ctrl_transfer(0x21,0x09,0,0,[0x02,0x10,0x00,0x00,0x00,0x00,0x00,0x00])

    
