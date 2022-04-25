#################################################################
#						SOCKET NETWORK							#
#################################################################

import time
import random
import socket
import threading
import sys

# Config and Create connection
BUFFER_SIZE = 4096

commands = dict(
    Disconnect=("!DISCONNECT", "!Disconnect", "!disconnect"),
)

# Server config
SERVER_IP = "178.128.121.175"
SERVER_PORT = 5050


def server_handler(client):
    """ Receive message from Sercer or Other client """
    while True:
        try:
            data = client.recv(BUFFER_SIZE)
        except Exception as e:
            print(f"\n [Server Error] : {e} \n")
            break

        if (not data) or (data.decode('utf-8') in commands['Disconnect']):
            print("\n [Server Response] Disconnected. \n")
            break

        print(
            f"\n [Recieve] Data : {data.decode('utf-8')} \nPress Enter to continue...")

    # Close Connection
    client.close()


def sender(msg: str) -> bool:
    """ Send message to Server or Other client """
    ping_broadcast = "!Ahoy"

    if msg == ping_broadcast:
        client.sendall(msg.encode('utf-8'))
    else:
        client.send(msg.encode('utf-8'))

    return True


# Create connection
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    client.connect((SERVER_IP, SERVER_PORT))
    print(f"\n [Connected] Hostname : {SERVER_IP} \n")
except socket.gaierror:
    print("Hostname Could Not Be Resolved !!!!")
    sys.exit()
except socket.error:
    print("Server not responding !!!!")
    sys.exit()


# Start threding server_handler
task = threading.Thread(target=server_handler, args=(client,))
task.start()


# try:
#     while True:
#         # message = input("[Send message] : ")
#         message = random.choice(['sit_foor', 'laying', 'a', 'b', 'c'])

#         success = sender(message)

#         # time.sleep(.00002)	# 0.02 ms
#         time.sleep(.14)  # 140 ms

#         if not success:
#             print("\n [FAILED!] Cannot send message \n")

#         if message in commands['Disconnect']:
#             print("\n [DISCONNECT!] \n")
#             break
# except KeyboardInterrupt:
#     print("Exiting Program !!!!")
#     client.send(commands['Disconnect'][0].encode('utf-8'))
# finally:
#     print("\n [Client Close!] Close Connection... \n")
#     client.close()
#     sys.exit()
