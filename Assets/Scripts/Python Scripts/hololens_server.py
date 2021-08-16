import asyncio
import base64
import time
import os

# Bind the socket to the port
#server_address = ('localhost', 11000)
#server_address = ('192.168.1.2', 11002)
#erver_address = ('129.169.34.183', 11002)
#server_address = ('192.168.1.111', 11003)
#server_address = ('localhost', 11003)

class Server:
    def __init__(self, ip_addr, port, buff_size):
        self.server_addr = ip_addr
        self.port = port

        self.buffer_size = buff_size
        self.image_data_buffer = []
        self.image_data_bytes = []
        self.image_data_bytes_recvd = 0


    async def tcp_echo_client(self, message, loop):
        reader, writer = await asyncio.open_connection(self.server_addr, self.port, loop=loop)

        print('Send: %r' % message)
        writer.write(message.encode())

        while True:
            try:
                data = await asyncio.wait_for(reader.read(self.buffer_size), timeout=15)
                #print('Received: %r' % data.decode())
                self.image_data_bytes.append(data)
                self.image_data_bytes_recvd += len(data)
                print('Bytes received from Hololens: %d' % len(data))
            except asyncio.TimeoutError:
                print('Timeout!')
                break

        print('Closing socket')
        writer.close()
        self.write_image_file()


    def write_image_file(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        directory = dir_path + "\\input_images\\"

        img_meta_data = ""
        timestamp = ""

        print('Image data bytes received: %d' % self.image_data_bytes_recvd)
        self.image_data_concat = b''.join(self.image_data_bytes)
        idx = self.image_data_concat.index(b'/ffff/')
        print(idx)
        self.image_data_buffer = self.image_data_concat[:idx]
        img_meta_data = self.image_data_concat[idx+6:].decode()

        t = time.time()
        timestamp = str(round(t))

        img_filename = directory + '\\context_img_%s.png' % timestamp
        print('Saving img to %s' % img_filename)
        with open(img_filename, "wb") as fh:
            fh.write(base64.decodebytes(self.image_data_buffer))
            fh.close()
        
        log_filename = directory + '\\context_img_%s.log' % timestamp                      
        print('Saving img meta data to %s' % log_filename)
        with open(log_filename, "wb") as fh:
            fh.write(img_meta_data.encode())
            fh.close()
        
        buffer_filename = directory + '\\context_img_buff_%s.log' % timestamp
        print('Saving img buffer to %s' % buffer_filename)
        with open(buffer_filename, "wb") as fh:
            fh.write(self.image_data_buffer)
            fh.close()


def main():
    server = Server('169.254.155.249', 9090, 10800)
    message = 'Starting up Hololens socket...'
    loop = asyncio.get_event_loop()
    server.tcp_echo_client(message, loop)
    loop.run_until_complete(server.tcp_echo_client(message, loop))
    loop.close()


if __name__ == "__main__":
    main()
