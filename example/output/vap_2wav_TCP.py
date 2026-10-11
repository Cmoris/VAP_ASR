#!/usr/bin/env python3
"""
This script is an example of how to use the VapGPT model with two WAV files.
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
    output_transmitter = MaaiOutput.TcpTransmitter(ip="127.0.0.1", port=50008, mode="vap")
    output_transmitter.start_server()

    # Receive the VAP result from the server
    output_receiver = MaaiOutput.TcpReceiver(ip="127.0.0.1", port=50008, mode="vap")
    output_receiver.start()

    output = MaaiOutput.ConsoleBar(bar_type="balance")

    maai = Maai(
        mode="vap",
        lang="jp",
        frame_rate=10,
        context_len_sec=5,
        audio_ch1=wav1,
        audio_ch2=wav2,
        device="cpu"
    )

    maai.start()

    while True:
        result = maai.get_result()
        # Send the result
        output_transmitter.update(result)
        # Receive the VAP result from the server
        result = output_receiver.get_result()
        # Update the result
        output.update(result)
        
if __name__ == "__main__":
    test()