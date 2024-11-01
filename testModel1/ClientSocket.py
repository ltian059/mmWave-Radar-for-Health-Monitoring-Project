import socket
import time
import ipaddress
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import struct
import os
# Define folder to watch
folder_to_watch = r"D:\python-learn\healthProjectSocket"  # Replace with the path to the folder you want to monitor
# Define the server's IPv6 address and port
HOST = '2607:fea8:bcc3:5d00:4238:c4ab:f17e:6909'  # Replace with the server's actual IPv6 address
PORT = 65432

def is_file_stable(file_path, wait_time=0.5):
    """Check if the file size remains stable for a certain period, indicating it is fully generated."""
    initial_size = os.path.getsize(file_path)
    time.sleep(wait_time)
    current_size = os.path.getsize(file_path)
    return initial_size == current_size

# Handler class to process new files
class NewFileHandler(FileSystemEventHandler):
    def __init__(self, client_socket):
        self.client_socket = client_socket

    def on_created(self, event):
        # Check if it's a file
        if not event.is_directory:
            file_path = event.src_path
            file_name = os.path.basename(file_path)

            # Wait until the file is fully generated
            if not is_file_stable(file_path):
                print(f"File '{file_name}' is not yet stable. Waiting for it to be fully generated...")
            while not is_file_stable(file_path):
                time.sleep(0.5)  # Poll until the file is stable

            file_size = os.path.getsize(file_path)
            print(f"New file detected: {file_path} (Size: {file_size} bytes)")

            # Read and send the file content
            try:
                # Send metadata: file name length, file name, and file size
                self.client_socket.sendall(struct.pack('!I', len(file_name)))
                self.client_socket.sendall(file_name.encode())
                self.client_socket.sendall(struct.pack('!Q', file_size))  # 8-byte file size

                # data Send file content in chunks
                with open(file_path, 'rb') as file:
                    while True:
                        chunk = file.read(4096)  # Send in chunks of 4096 bytes
                        if not chunk:
                            break
                        self.client_socket.sendall(chunk)
                    print(f"Sent file {event.src_path}.")
            except Exception as e:
                print(f"Failed to send file: {e}")

def check_ip_version(ip):
    try:
        # Attempt to create an IPv4Address object
        if isinstance(ipaddress.ip_address(ip), ipaddress.IPv4Address):
            return "IPv4"
        # If successful, it’s an IPv4 address
    except ValueError:
        pass  # If ValueError is raised, it's not a valid IPv4 address

    try:
        # Attempt to create an IPv6Address object
        if isinstance(ipaddress.ip_address(ip), ipaddress.IPv6Address):
            return "IPv6"
        # If successful, it’s an IPv6 address
    except ValueError:
        pass  # If ValueError is raised, it's not a valid IPv6 address

    return "Invalid IP address"  # If neither worked, it's not a valid IP

# Create an IPv6 TCP socket
socket_type = ""
is_ipv4 = True
if "IPv4" == check_ip_version(HOST):
    socket_type = socket.AF_INET
elif "IPv6" == check_ip_version(HOST):
    socket_type = socket.AF_INET6
    is_ipv4 = False

with socket.socket(socket_type, socket.SOCK_STREAM) as client_socket:
    if is_ipv4:
        client_socket.connect((HOST, PORT))
    else:
        client_socket.connect((HOST, PORT, 0, 0))  # Connect to the IPv6 server

    print("Connected to server successfully!")
    # Set up watchdog observer to watch folder for new files
    event_handler = NewFileHandler(client_socket)
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Stopped folder monitoring.")
        print("Client connection interrupted")
    observer.join()

