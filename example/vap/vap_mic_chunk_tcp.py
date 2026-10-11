#!/usr/bin/env python3
"""
VAPモデルのテストスクリプト
マイク（TCP経由）と0入力をチャンク入力してVAPモデルの動作を確認します

Test script for the VAP model.
This script verifies the operation of the VAP model by inputting chunks from a microphone via TCP and a zero-filled array.
"""

import sys
import os

import queue
import threading

import numpy as np
import pyaudio
import time

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput


def run_maai_with_mic_chunks():
    CHUNK_SIZE = 160

    tcp_chunk = MaaiInput.TcpChunk(server_ip="127.0.0.1", port=5000)  # マイク入力用
    zero_input = MaaiInput.Chunk()  # ゼロ配列入力用

    tcp = MaaiInput.Tcp(port=5000)

    # Start the receiver and transmitter
    tcp.start_server()
    time.sleep(3)  # Wait for the server to start
    tcp_chunk.start()

    # MaAIの初期化
    maai = Maai(
        mode="vap",
        lang="jp",
        frame_rate=10,
        audio_ch1=tcp,
        audio_ch2=zero_input,
        device="cpu",
    )
    maai.start()
    maai_output_bar = MaaiOutput.ConsoleBar(bar_type="balance")

    # オーディオチャンクを格納するキュー
    audio_queue = queue.Queue()
    stream_active = threading.Event()
    stream_active.set()

    # ゼロ配列チャンク（チャンネル2用）
    zeros_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)

    # PyAudioコールバック関数
    audio_interface = pyaudio.PyAudio()

    def audio_callback(in_data, frame_count, time_info, status):
        if stream_active.is_set():
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_queue.put(audio_data.copy())
        return (None, pyaudio.paContinue)

    try:
        # PyAudioストリームを開始
        stream = audio_interface.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=audio_callback,
        )

        stream.start_stream()

        while stream.is_active():
            try:
                # キューから音声チャンクを取得（タイムアウト付き）
                mic_chunk = audio_queue.get(timeout=1.0)

                # チャンクサイズの確認
                if len(mic_chunk) != CHUNK_SIZE:
                    continue

                # チャンクデータを更新
                tcp_chunk.put_chunk(mic_chunk)
                zero_input.put_chunk(zeros_chunk)

                # 結果をノンブロッキングで取得して表示
                result = maai.result_dict_queue.get_nowait()
                if result:
                    maai_output_bar.update(result)

            except queue.Empty:
                continue
    finally:
        # ストリームの停止とクリーンアップ
        stream_active.clear()
        if "stream" in locals():
            stream.stop_stream()
            stream.close()
        audio_interface.terminate()


if __name__ == "__main__":
    run_maai_with_mic_chunks()
