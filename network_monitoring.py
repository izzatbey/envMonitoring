import time
import os
from scapy.all import sniff
import psutil
from collections import defaultdict
from threading import Thread
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import plotly.io as pio

# Get all network adapter's MAC addresses
all_macs = {iface.address for iface_list in psutil.net_if_addrs().values() for iface in iface_list if iface.family == psutil.AF_LINK}
# A dictionary to map each connection to its corresponding process ID (PID)
connection2pid = {}
# A dictionary to map each process ID (PID) to total Upload (0) and Download (1) traffic
pid2traffic = defaultdict(lambda: [0, 0])
# The global Pandas DataFrame that's used to track previous traffic stats
global_df = None
# Global boolean for status of the program
is_program_running = True

# Plotly graph initialization
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Upload Speed', 'Download Speed'))
fig.update_layout(height=600, width=800, title_text="Network Traffic")
upload_trace = go.Scatter(x=[], y=[], mode='lines', name='Upload Speed')
download_trace = go.Scatter(x=[], y=[], mode='lines', name='Download Speed')
fig.add_trace(upload_trace, row=1, col=1)
fig.add_trace(download_trace, row=2, col=1)

def get_size(bytes):
    """
    Returns size of bytes in a nice format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024

def process_packet(packet):
    global pid2traffic
    try:
        # Get the packet source & destination IP addresses and ports
        packet_connection = (packet.sport, packet.dport)
    except (AttributeError, IndexError):
        # Sometimes the packet does not have TCP/UDP layers, we just ignore these packets
        pass
    else:
        # Get the PID responsible for this connection from our `connection2pid` global dictionary
        packet_pid = connection2pid.get(packet_connection)
        if packet_pid:
            if packet.src in all_macs:
                # The source MAC address of the packet is our MAC address
                # so it's an outgoing packet, meaning it's upload
                pid2traffic[packet_pid][0] += len(packet)
            else:
                # Incoming packet, download
                pid2traffic[packet_pid][1] += len(packet)

def get_connections():
    """A function that keeps listening for connections on this machine 
    and adds them to `connection2pid` global variable"""
    global connection2pid
    while is_program_running:
        # Using psutil, we can grab each connection's source and destination ports
        # and their process ID
        for c in psutil.net_connections():
            if c.laddr and c.raddr and c.pid:
                # If local address, remote address and PID are in the connection
                # add them to our global dictionary
                connection2pid[(c.laddr.port, c.raddr.port)] = c.pid
                connection2pid[(c.raddr.port, c.laddr.port)] = c.pid
        # Sleep for a second, feel free to adjust this
        time.sleep(1)

def print_pid2traffic():
    global global_df
    # Initialize the list of processes
    processes = []
    for pid, traffic in pid2traffic.items():
        # `pid` is an integer that represents the process ID
        # `traffic` is a list of two values: total Upload and Download size in bytes
        try:
            # Get the process object from psutil
            p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            # If process is not found, simply continue to the next PID for now
            continue
        # Get the name of the process, such as chrome.exe, etc.
        name = p.name()
        # Get the time the process was spawned
        try:
            create_time = datetime.fromtimestamp(p.create_time())
        except OSError:
            # System processes, using boot time instead
            create_time = datetime.fromtimestamp(psutil.boot_time())
        # Construct our dictionary that stores process info
        process = {
            "pid": pid, "name": name, "create_time": create_time, "Upload": traffic[0],
            "Download": traffic[1],
        }
        try:
            # Calculate the upload and download speeds by simply subtracting the old stats from the new stats
            process["Upload Speed"] = traffic[0] - global_df.at[pid, "Upload"]
            process["Download Speed"] = traffic[1] - global_df.at[pid, "Download"]
        except (KeyError, AttributeError):
            # If it's the first time running this function, then the speed is the current traffic
            # You can think of it as if old traffic is 0
            process["Upload Speed"] = traffic[0]
            process["Download Speed"] = traffic[1]
        # Append the process to our processes list
        processes.append(process)
    # Construct our Pandas DataFrame
    df = pd.DataFrame(processes)
    try:
        # Set the PID as the index of the dataframe
        df = df.set_index("pid")
        # Sort by column, feel free to edit this column
        df.sort_values("Download", inplace=True, ascending=False)
    except KeyError as e:
        # When dataframe is empty
        pass
    # Make another copy of the dataframe just for fancy printing
    printing_df = df.copy()
    try:
        # Apply the function get_size to scale the stats like '532.6KB/s', etc.
        printing_df["Download"] = printing_df["Download"].apply(get_size)
        printing_df["Upload"] = printing_df["Upload"].apply(get_size)
        printing_df["Download Speed"] = printing_df["Download Speed"].apply(get_size).apply(lambda s: f"{s}/s")
        printing_df["Upload Speed"] = printing_df["Upload Speed"].apply(get_size).apply(lambda s: f"{s}/s")
    except KeyError as e:
        # When dataframe is empty again
        pass
    # Clear the screen based on your OS
    os.system("cls") if "nt" in os.name else os.system("clear")
    # Print our dataframe
    print(printing_df.to_string())
    # Update the global df to our dataframe
    global_df = df
    # Update Plotly graph
    update_plotly_graph(df)

def update_plotly_graph(df):
    global fig, upload_trace, download_trace
    current_time = datetime.now().strftime("%H:%M:%S")
    try:
        upload_speed = df["Upload Speed"].sum()
        download_speed = df["Download Speed"].sum()
    except KeyError:
        upload_speed = 0
        download_speed = 0
    
    # Update the data in traces
    upload_trace = fig.data[0]
    download_trace = fig.data[1]
    upload_trace.x = list(upload_trace.x) + [current_time]
    upload_trace.y = list(upload_trace.y) + [upload_speed]
    download_trace.x = list(download_trace.x) + [current_time]
    download_trace.y = list(download_trace.y) + [download_speed]

    # Update the figure
    fig.update_traces(upload_trace, selector=dict(name='Upload Speed'))
    fig.update_traces(download_trace, selector=dict(name='Download Speed'))

def print_stats():
    """Simple function that keeps printing the stats"""
    while is_program_running:
        time.sleep(1)
        print_pid2traffic()

if __name__ == "__main__":
    # Start the printing thread
    printing_thread = Thread(target=print_stats)
    printing_thread.start()
    # Start the get_connections() function to update the current connections of this machine
    connections_thread = Thread(target=get_connections)
    connections_thread.start()

    # Start sniffing
    print("Started sniffing")
    sniff(prn=process_packet, store=False, timeout=60)
    # Setting the global variable to False to exit the program
    is_program_running = False

    # Show the figure
    fig.show()

    # Save the figure as a PNG file
    pio.write_image(fig, 'network_traffic.png')
    print("Graph saved as network_traffic.png")
