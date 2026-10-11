"""
This script is an example of how to use the VapGPT model with prompt control.
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput

def test_vap():

    output = MaaiOutput.ConsoleBar()

    print("Initializing MaAI with VapGPT model and prompt control...")
    maai = Maai(
        mode="vap_prompt",
        lang="jp",
        frame_rate=10,
        context_len_sec=20,
        audio_ch1=MaaiInput.Mic(),
        audio_ch2=MaaiInput.Zero(),
        device="cpu"
    )
    print("MaAI initialized.")

    # Set prompts for channel 1 and channel 2
    maai.set_prompt_ch1("テンポよく発話し、相手の発言が終わるとすぐに返答してください。発言回数を多めに、会話をリードするようにしてください。")
    maai.set_prompt_ch2("発話前に少し間を取り、考えてから丁寧に話し始めてください。応答は急がず、落ち着いたテンポを意識してください。")

    maai.start()

    while True:
        result = maai.get_result()
        output.update(result)
        
if __name__ == "__main__":
    try:
        test_vap()
    except KeyboardInterrupt:
        print("Ending the test script.")