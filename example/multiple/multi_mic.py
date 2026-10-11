"""
Run multiple Maai models in parallel from a single microphone input,
sharing one audio encoder via :class:`MaaiMultiple` for efficiency.

This example combines:
- Turn-taking (``vap``) prediction
- Backchannel (``bc``) prediction
- Nodding (``nod``) prediction
"""

import sys
import os

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import MaaiMultiple, MaaiInput, MaaiOutput


def test():

    mic = MaaiInput.Mic()
    zero = MaaiInput.Zero()

    maai_multi = MaaiMultiple(
        configs=[
            {"mode": "vap", "lang": "jp"},
            {"mode": "bc", "lang": "jp"},
            {"mode": "nod", "lang": "jp"},
        ],
        audio_ch1=mic,
        audio_ch2=zero,
        frame_rate=10,
        device="cpu",
        model_type="normal",
    )

    # ConsoleBar automatically detects the MaaiMultiple result format and
    # renders one section per sub-model.
    output = MaaiOutput.ConsoleBar(bar_type="balance")

    maai_multi.start()

    while True:
        result = maai_multi.get_result()
        output.update(result)

if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("Ending the test script.")
