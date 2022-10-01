# Copyright (c) 2022 Savoir-faire Linux Inc.
# This code is licensed under MIT license
import sounddevice as sd
import asyncio
import numpy as np

async def inputstream_generator(channels=1, samplerate=16000, **kwargs):
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(callback=callback, channels=channels, samplerate=samplerate, **kwargs)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def record_audio_blocks(block_size, slide, samplerate, **kwargs):
    audio_buf = np.ndarray(1, dtype=np.float32)
    async for indata, status in inputstream_generator(samplerate=samplerate, **kwargs):
        if status:
            print(status)
        audio_buf = np.concatenate((audio_buf, indata[:,0]))
        if len(audio_buf) > block_size:
            copy = audio_buf[-block_size:]
            audio_buf = audio_buf[-slide:]
            yield copy
