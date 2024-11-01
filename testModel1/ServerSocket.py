import socket
import ipaddress
import os
import struct
import select
import Configuration

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

SAVE_DIR = Configuration.MODEL_DATA_INPUT_FOLDER
os.makedirs(SAVE_DIR, exist_ok=True)

def reliable_recv(conn, n):
    """ Helper function to reliably receive `n` bytes from the connection """
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None  # Connection closed
        data += packet
    return data

def get_data_from_socket(host, port, timeout = 60):
    is_ipv4 = True
    socket_type = ""
    if "IPv4" == check_ip_version(host):
        socket_type = socket.AF_INET
    elif "IPv6" == check_ip_version(host):
        is_ipv4 = False
        socket_type = socket.AF_INET6
    with socket.socket(socket_type, socket.SOCK_STREAM) as server_socket:
        # Allow address reuse to avoid "Address already in use" errors
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if is_ipv4:
            server_socket.bind((host, port))
        else:
            server_socket.bind((host, port, 0, 0))  # Bind to a IPv6 address

        server_socket.listen()
        server_socket.setblocking(False)  # Set non-blocking mode
        print("Waiting for client connection...")
        while True:
            try:
                # Use select to wait for a connection or time out after `timeout`
                readable, _, _ = select.select([server_socket], [], [], timeout)
                if readable:
                    conn, addr = server_socket.accept()
                    conn.settimeout(timeout)  # Set timeout for each connection
                    with conn:
                        print(f"Connected to client address：{addr}")
                        while True:
                            # Receive metadata: file name length
                            name_len_data = conn.recv(4)
                            if not name_len_data:
                                break  # No data, close the connection
                            name_len = struct.unpack('!I', name_len_data)[0]  # Get name length
                            file_name_data = reliable_recv(conn, name_len)
                            if not file_name_data:
                                break
                            file_name = file_name_data.decode('utf-8', errors='ignore')  # Ig

                            # Receive file size
                            file_size_data = reliable_recv(conn, 8)
                            if not file_size_data:
                                break
                            file_size = struct.unpack('!Q', file_size_data)[0]

                            print(f"Receiving file '{file_name}' of size {file_size} bytes...")
                            # Define full path for the file
                            file_path = os.path.join(SAVE_DIR, file_name)

                            # Save received file
                            with open(file_path, 'wb') as file:
                                received_size = 0
                                while received_size < file_size:
                                    chunk = conn.recv(min(4096, file_size - received_size))
                                    if not chunk:
                                        break
                                    file.write(chunk)
                                    received_size += len(chunk)
                            print(f"File '{file_name}' received successfully, saved as '{file_name}'.")
                else:
                    # No activity detected, continuing to check
                    print("No new connection detected. Still in hibernate mode.")
            except KeyboardInterrupt:
                print("Server manually interrupted. Shutting down.")
                break  # Exit the loop to stop the server


get_data_from_socket("::", 65432)