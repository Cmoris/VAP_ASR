#!/usr/bin/env python3
"""
This script is an example of how to use the VapGPT model with two WAV files.
Also, it demonstrates how to set up a TCP server for output.
"""

import sys
import os

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput

def test():

    wav1 = MaaiInput.Wav(wav_file_path="../wav_sample/jpn_inoue_16k.wav")
    wav2 = MaaiInput.Wav(wav_file_path="../wav_sample/jpn_sumida_16k.wav")

    # Output server
    tcp_out = MaaiOutput.TcpTransmitter(ip="127.0.0.1", port=50008, mode="vap")
    tcp_out.start_server()

    # Receive the VAP result from the server
    tcp_recv = MaaiOutput.TcpReceiver(ip="127.0.0.1", port=50008, mode="vap")
    tcp_recv.start()

    output = MaaiOutput.GuiPlot()

    maai = Maai(
        mode="vap",
        lang="jp",
        frame_rate=10,
        audio_ch1=wav1,
        audio_ch2=wav2,
        device="cpu"
    )

    maai.start()

    while True:
        # Send the result to the output server
        result = maai.get_result()
        tcp_out.update(result)

        # Receive the VAP result from the server
        result = tcp_recv.get_result()
        output.update(result)

if __name__ == "__main__":
    test()
