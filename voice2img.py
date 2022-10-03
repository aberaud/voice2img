#!/usr/bin/env python
# Copyright (c) 2022 Savoir-faire Linux Inc.
# This code is licensed under MIT license
import argparse
from pytorch_lightning import seed_everything
import asyncio
import concurrent.futures
import aiohttp
from aiohttp import web
import uuid

from image_generator import ImageGenerator
from speech_recognizer import SpeechRecognizer
from audio_input import record_audio_blocks

speech_executor = concurrent.futures.ThreadPoolExecutor(1)
imggen_executor = concurrent.futures.ThreadPoolExecutor(1)

BUF_SIZE = 8*SpeechRecognizer.SAMPLE_RATE
SLIDE = BUF_SIZE//4

queue = asyncio.Queue()
image_generator = ImageGenerator()
speech_recognizer = SpeechRecognizer()
connected_clients: set[web.WebSocketResponse] = set()
cached_result = None


def load_model(opt):
    image_generator.init(opt)


def load_speech(opt):
    speech_recognizer.init()


def generate_images(prompt):
    return image_generator.generate_images(prompt)


def recognize_speech(a):
    return speech_recognizer.process_audio(a)


async def index(request):
    return web.FileResponse('./index.html')


async def send_to_client(socket: web.WebSocketResponse, input, images=()):
    try:
        await socket.send_json(input)
        for img in images:
            await socket.send_bytes(img)
    except Exception as e:
        print(f'Error sending to client: {e}')


async def audio_to_text():
    print('Starting audio recording...')
    async for audio_frame in record_audio_blocks(block_size=BUF_SIZE, slide=SLIDE, samplerate=SpeechRecognizer.SAMPLE_RATE):
        lang, result = await asyncio.get_running_loop().run_in_executor(speech_executor, recognize_speech, audio_frame)
        if result:
            data = {'id': str(uuid.uuid4()), 'lang': lang, 'text': result}
            await asyncio.gather(
                queue.put(data),
                *[send_to_client(ws, data) for ws in connected_clients]
            )


async def text_to_image():
    global cached_result
    while True:
        task = await queue.get()
        result, images = await asyncio.get_running_loop().run_in_executor(imggen_executor, generate_images, task)
        result['n'] = len(images) if images else 0
        cached_result = result, images
        await asyncio.gather(*[send_to_client(ws, result, images) for ws in connected_clients])
        queue.task_done()


async def websocket_handler(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print(f'WebSocket connection with {request.remote} ({" ".join(request.headers["User-Agent"].split()[-2:])}) opened')

    # Send initial cached data
    if cached_result:
        await send_to_client(ws, *cached_result)
    connected_clients.add(ws)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                await ws.close()
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print(f'WebSocket connection with {request.remote} closed with exception {ws.exception()}')
    connected_clients.remove(ws)
    print(f'WebSocket connection with {request.remote} closed')
    return ws


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)

    app = web.Application()
    app.add_routes([web.get('/', index)])
    app.add_routes([web.get('/ws', websocket_handler)])

    loop = asyncio.get_running_loop()
    # Load models in their executor
    await loop.run_in_executor(imggen_executor, load_model, opt)
    await loop.run_in_executor(speech_executor, load_speech, opt)

    # Start processing
    await asyncio.gather(
        audio_to_text(),
        text_to_image(),
        web._run_app(app)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
