"""
This script is an example of using a single microphone via TCP with the VapGPT model.
"""

import sys
import os
import time

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput

def test():

    # Use the default mic via TCP
    tcp_mic = MaaiInput.TcpMic(server_ip="127.0.0.1", port=5000)
    tcp = MaaiInput.Tcp(port=5000)

    # Start the receiver and transmitter
    tcp.start_server()
    time.sleep(3)  # Wait for the server to start
    tcp_mic.start()

    # Use zero signals for the second channel
    zero = MaaiInput.Zero()

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="vap",
        lang="jp",
        frame_rate=10,
        audio_ch1=tcp,
        audio_ch2=zero,
        device="cpu"
    )

    maai.start()

    while True:
        result = maai.get_result()
        output.update(result)
        
if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("Ending the test script.")