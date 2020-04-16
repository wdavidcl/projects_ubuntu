import serial
import struct

serialPort = serial.Serial( port = "/dev/ttyUSB0", 
                            baudrate=115200,
                            bytesize=8, 
                            timeout=500, 
                            stopbits=serial.STOPBITS_ONE)

serialString = ""                           # Used to hold data coming over UART
w = 27.9
u = 0
serialPort.write(b"0.0\r\n")

while True:
    
    #serialPort.write(str(u).encode('utf-8'))
    # Wait until there is data waiting in the serial buffer
    if(serialPort.in_waiting > 0):
        # Read data out of the buffer until a carraige return / new line is found
        serialString = serialPort.readline()
        # Print the contents of the serial data
        try:
            y = float(serialString.decode('Ascii'))
        except:
            y=0
        #print(y)
        # Tell the device connected over the serial port that we recevied the data!
        # The b at the beginning is used to indicate bytes!
        u = -1.4048*y+2.4048*w
        aux=str(u).encode('Ascii')
        serialPort.write(b"\r\n"+aux)
        print(w,u,y)
        
        
