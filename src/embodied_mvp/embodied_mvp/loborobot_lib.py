"""Makerobo LOBOROBOT control library — vendored from kit's ~/makerobo_code/LOBOROBOT.py.

Original: 湖南创乐博智能科技 (Makerobo), author zhulin, V2.0.
Only change vs vendor file: `import smbus` -> smbus2 fallback, so it runs in the
RoboStack conda env (conda has no system python3-smbus). smbus2's SMBus is API
compatible for write_byte_data / read_byte_data.

Hardware: single PCA9685 @ I2C 0x40 drives 4 motors (ch 0-8,11) and 2 servos
(ch 9 tilt, ch 10 pan). Motor D direction also uses GPIO 24/25 via gpiozero.
"""
import time
import math

try:
    from smbus2 import SMBus
except ImportError:  # fall back to system python3-smbus
    from smbus import SMBus

from gpiozero import LED

Dir = [
    'forward',
    'backward',
]


class PCA9685:
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09

    def __init__(self, address, debug=False):
        self.bus = SMBus(1)
        self.address = address
        self.debug = debug
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)

    def read(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def setPWMFreq(self, freq):
        prescaleval = 25000000.0
        prescaleval /= 4096.0
        prescaleval /= float(freq)
        prescaleval -= 1.0
        prescale = math.floor(prescaleval + 0.5)
        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10
        self.write(self.__MODE1, newmode)
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)

    def setDutycycle(self, channel, pulse):
        self.setPWM(channel, 0, int(pulse * (4096 / 100)))

    def setLevel(self, channel, value):
        if value == 1:
            self.setPWM(channel, 0, 4095)
        else:
            self.setPWM(channel, 0, 0)


class LOBOROBOT():
    def __init__(self):
        self.PWMA = 0
        self.AIN1 = 2
        self.AIN2 = 1
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4
        self.PWMC = 6
        self.CIN2 = 7
        self.CIN1 = 8
        self.PWMD = 11
        self.DIN1 = 25
        self.DIN2 = 24

        self.pwm = PCA9685(0x40, debug=False)
        self.pwm.setPWMFreq(50)
        self.motorD1 = LED(self.DIN1)
        self.motorD2 = LED(self.DIN2)

    def MotorRun(self, motor, index, speed):
        if speed > 100:
            return
        if motor == 0:
            self.pwm.setDutycycle(self.PWMA, speed)
            if index == Dir[0]:
                self.pwm.setLevel(self.AIN1, 0)
                self.pwm.setLevel(self.AIN2, 1)
            else:
                self.pwm.setLevel(self.AIN1, 1)
                self.pwm.setLevel(self.AIN2, 0)
        elif motor == 1:
            self.pwm.setDutycycle(self.PWMB, speed)
            if index == Dir[0]:
                self.pwm.setLevel(self.BIN1, 1)
                self.pwm.setLevel(self.BIN2, 0)
            else:
                self.pwm.setLevel(self.BIN1, 0)
                self.pwm.setLevel(self.BIN2, 1)
        elif motor == 2:
            self.pwm.setDutycycle(self.PWMC, speed)
            if index == Dir[0]:
                self.pwm.setLevel(self.CIN1, 1)
                self.pwm.setLevel(self.CIN2, 0)
            else:
                self.pwm.setLevel(self.CIN1, 0)
                self.pwm.setLevel(self.CIN2, 1)
        elif motor == 3:
            self.pwm.setDutycycle(self.PWMD, speed)
            if index == Dir[0]:
                self.motorD1.off()
                self.motorD2.on()
            else:
                self.motorD1.on()
                self.motorD2.off()

    def MotorStop(self, motor):
        if motor == 0:
            self.pwm.setDutycycle(self.PWMA, 0)
        elif motor == 1:
            self.pwm.setDutycycle(self.PWMB, 0)
        elif motor == 2:
            self.pwm.setDutycycle(self.PWMC, 0)
        elif motor == 3:
            self.pwm.setDutycycle(self.PWMD, 0)

    def t_up(self, speed, t_time):
        self.MotorRun(0, 'forward', speed)
        self.MotorRun(1, 'forward', speed)
        self.MotorRun(2, 'forward', speed)
        self.MotorRun(3, 'forward', speed)
        time.sleep(t_time)

    def t_down(self, speed, t_time):
        self.MotorRun(0, 'backward', speed)
        self.MotorRun(1, 'backward', speed)
        self.MotorRun(2, 'backward', speed)
        self.MotorRun(3, 'backward', speed)
        time.sleep(t_time)

    def moveLeft(self, speed, t_time):
        self.MotorRun(0, 'backward', speed)
        self.MotorRun(1, 'forward', speed)
        self.MotorRun(2, 'forward', speed)
        self.MotorRun(3, 'backward', speed)
        time.sleep(t_time)

    def moveRight(self, speed, t_time):
        self.MotorRun(0, 'forward', speed)
        self.MotorRun(1, 'backward', speed)
        self.MotorRun(2, 'backward', speed)
        self.MotorRun(3, 'forward', speed)
        time.sleep(t_time)

    def turnLeft(self, speed, t_time):
        self.MotorRun(0, 'backward', speed)
        self.MotorRun(1, 'forward', speed)
        self.MotorRun(2, 'backward', speed)
        self.MotorRun(3, 'forward', speed)
        time.sleep(t_time)

    def turnRight(self, speed, t_time):
        self.MotorRun(0, 'forward', speed)
        self.MotorRun(1, 'backward', speed)
        self.MotorRun(2, 'forward', speed)
        self.MotorRun(3, 'backward', speed)
        time.sleep(t_time)

    def t_stop(self, t_time):
        self.MotorStop(0)
        self.MotorStop(1)
        self.MotorStop(2)
        self.MotorStop(3)
        time.sleep(t_time)

    def set_servo_pulse(self, channel, pulse):
        pulse_length = 1000000
        pulse_length //= 60
        pulse_length //= 4096
        pulse *= 1000
        pulse //= pulse_length
        self.pwm.setPWM(channel, 0, pulse)

    def set_servo_angle(self, channel, angle):
        angle = 4096 * ((angle * 11) + 500) / 20000
        self.pwm.setPWM(channel, 0, int(angle))
