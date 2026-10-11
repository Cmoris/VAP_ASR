#!/usr/bin/env python3
"""
This script is an example of how to use the VapGPT model with a microphone and a WAV file.
"""

import sys
import os

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput

def test():

    mic = MaaiInput.Mic()
    wav = MaaiInput.Wav(wav_file_path="../wav_sample/jpn_inoue_16k.wav")

    output = MaaiOutput.GuiPlot()

    maai = Maai(
        mode="vap",
        lang="jp",
        frame_rate=10,
        audio_ch1=mic,
        audio_ch2=wav,
        device="cpu"
    )

    maai.start_process()

    while True:
        result = maai.get_result()
        output.update(result)
        
if __name__ == "__main__":
    test()