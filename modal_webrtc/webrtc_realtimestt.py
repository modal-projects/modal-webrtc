import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcSignalingServer

py_version = "3.12"
ld_libray_path = f"/usr/local/lib/python{py_version}/site-packages/nvidia/cudnn/lib"

video_processing_image = (
    modal.Image.debian_slim(python_version=py_version)  # matching ld path
    # update locale as required by onnx
    .env({"LD_LIBRARY_PATH": ld_libray_path})
    # install system dependencies
    .apt_install("python3-opencv", "ffmpeg","python3-dev","portaudio19-dev", "python3-opencv")
    # install Python dependencies
    .uv_pip_install(
        "aiortc",
        "fastapi",
        "huggingface-hub[hf_xet]",
        "onnxruntime-gpu",
        "opencv-python",
        "torch",
        "shortuuid",
        "RealtimeSTT",
        "aiologic",
    )
)

CACHE_VOLUME = modal.Volume.from_name("webrtc-realtimestt-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")
cache = {CACHE_PATH: CACHE_VOLUME}

app = modal.App("webrtc-realtime-stt")

# default text_detected function, will be replaced with something that sends the message out to our system
def text_detected(text):
    text = f"\r{text}"
    print(f"\r{text}", flush=True, end='')


with video_processing_image.imports():
    import asyncio
    import threading
    import aiologic
    from aiortc import MediaStreamTrack
    from huggingface_hub import snapshot_download
    import RealtimeSTT

@app.cls(
    image=video_processing_image,
    gpu=["L40S"],
    volumes=cache,
    secrets=[modal.Secret.from_name("turn-credentials")],
    # region="us-east",  # set to your region
    min_containers=1,
    # buffer_containers=1,
    # scaledown_window=2,
    # max_inputs=1,
)
@modal.concurrent(max_inputs=5)
class RealtimeSTTPeer(ModalWebRtcPeer):

    async def initialize(self):

        self.recorder_config = {
            'download_root': CACHE_PATH,
            'spinner': False,
            'use_microphone': False,
            'model': 'large-v3',
            'language': 'en',
            'silero_sensitivity': 0.7,
            'silero_use_onnx': True,
            'silero_deactivity_detection': True,
            'post_speech_silence_duration': 0.5,
            'early_transcription_on_silence': 0.250,
            'min_length_of_recording': 0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'realtime_model_type': 'tiny.en',
            'on_realtime_transcription_stabilized': text_detected,      
            'initial_prompt': 'Add periods only for complete sentences. Use ellipsis (...) for unfinished thoughts or unclear endings. Examples: \n- Complete: I went to the store.\n- Incmplete: I think it was...'

        }

        # download models and pass paths to threads to avoid hf_hub + tqdm thread safety issues when
        # we load the models via hf_hub inside the threads
        if not os.path.exists(CACHE_PATH / f"models--Systran--faster-whisper-{self.recorder_config['model']}"):
        
            self.recorder_config['model'] = snapshot_download(
                f"Systran/faster-whisper-{self.recorder_config['model']}", 
                local_files_only=False, 
                cache_dir=CACHE_PATH,
            )
       
        if not os.path.exists(CACHE_PATH / f"models--Systran--faster-whisper-{self.recorder_config['realtime_model_type']}"):
            self.recorder_config['realtime_model_type'] = snapshot_download(
            f"Systran/faster-whisper-{self.recorder_config['realtime_model_type']}", 
            local_files_only=False, 
            cache_dir=CACHE_PATH,
        )

        self.audio_relays = {}
        self.transcription_to_datachannel_queues = {}
        self.transcription_to_subtitle_queues = {}
        self.stop_events = {}
        self.data_channels = {}
        self.send_transcription_to_datachannel_tasks = {}
    
    async def _start_transcription_threads(self):
        stt_model = RealtimeSTT.AudioToTextRecorder(**self.recorder_config)
        print("model loaded")

        audio_queue = aiologic.SimpleQueue()
        transcription_to_datachannel_queue = aiologic.SimpleQueue()
        transcription_to_subtitle_queue = aiologic.SimpleQueue()

        audio_relay = get_audio_relay(audio_queue)

        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=feed_audio_thread, 
            args=(
                stt_model, 
                audio_queue, 
                stop_event
            )
        )
        audio_thread.start()

        # Start the transcription_thread
        transcription_thread = threading.Thread(
            target=recorder_transcription_thread,
            args=(
                stt_model, 
                [transcription_to_subtitle_queue, transcription_to_datachannel_queue],
                stop_event,
            )
        )
        transcription_thread.start()

        print(f"RealtimeSTT initialized....")

        return audio_relay, transcription_to_datachannel_queue, transcription_to_subtitle_queue, stop_event

    async def initialize_peer(self, peer_id):
        (
            self.audio_relays[peer_id], 
            self.transcription_to_datachannel_queues[peer_id], 
            self.transcription_to_subtitle_queues[peer_id], 
            self.stop_events[peer_id],
        ) = await self._start_transcription_threads()

    async def setup_streams(self, peer_id: str):

        # keep us notified on connection state changes
        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if self.pcs[peer_id]:
                print(
                    f"Modal WebRTC Peer, {self.id}, connection state to {peer_id}: {self.pcs[peer_id].connectionState}"
                )


        # when we receive a track from the source peer
        # we create a processed track and add it to our stream
        # back to the source peer
        @self.pcs[peer_id].on("track")
        def on_track(track: MediaStreamTrack) -> None:
            print(
                f"Modal WebRTC Peer, {self.id}, received {track.kind} track from {peer_id}"
            )

            if track.kind == "video":
                self.pcs[peer_id].addTrack(get_subtitle_track(track, self.transcription_to_subtitle_queues[peer_id]))
                @track.on("ended")
                async def on_ended() -> None:
                    print(
                        f"Modal WebRTC Peer, {self.id}, incoming video track from {peer_id} ended"
                    )
                
            elif track.kind == "audio":

                self.audio_relays[peer_id].add_track(track)

                # keep us notified when the incoming track ends
                @track.on("ended")
                async def on_ended() -> None:
                    print(
                        f"Video Processor, {self.id}, incoming audio track from {peer_id} ended"
                    )
                    # self.stop_events[peer_id].set()
                    # await self.audio_relays[peer_id].stop()

                    print("Audio relay stopped")


        @self.pcs[peer_id].on("datachannel")
        def on_datachannel(channel):
            print("Data channel opened")
            self.data_channels[peer_id] = channel
            @self.data_channels[peer_id].on("open")
            def on_open():
                print("Data channel opened")


    async def run_streams(self, peer_id):

        await self.audio_relays[peer_id].start()
        # wait for data channel to be opened
        while not self.data_channels[peer_id] or self.data_channels[peer_id].readyState != "open":
            await asyncio.sleep(0.1)
        self.send_transcription_to_datachannel_tasks[peer_id] = asyncio.create_task(
            _send_transcription_to_client(self.transcription_to_datachannel_queues[peer_id], self.data_channels[peer_id])
        )
        
        print("Send transcription task started")

    async def exit(self, peer_id):

        # stop threads
        self.stop_events[peer_id].set()

        # stop audio relay
        if self.audio_relays[peer_id]:
            await self.audio_relays[peer_id].stop()

        # stop send transcription to datachannel task
        if self.send_transcription_to_datachannel_tasks[peer_id] is not None:
            self.send_transcription_to_datachannel_tasks[peer_id].cancel()
            
        print("Send transcription task cancelled")
        
    async def get_turn_servers(self, peer_id=None, msg=None) -> dict:
        creds = {
            "username": os.environ["TURN_USERNAME"],
            "credential": os.environ["TURN_CREDENTIAL"],
        }

        turn_servers = [
            {"urls": "stun:stun.relay.metered.ca:80"},  # STUN is free, no creds neeeded
            # for TURN, sign up for the free service here: https://www.metered.ca/tools/openrelay/
            {"urls": "turn:standard.relay.metered.ca:80"} | creds,
            {"urls": "turn:standard.relay.metered.ca:80?transport=tcp"} | creds,
            {"urls": "turn:standard.relay.metered.ca:443"} | creds,
            {"urls": "turns:standard.relay.metered.ca:443?transport=tcp"} | creds,
        ]

        return {"type": "turn_servers", "ice_servers": turn_servers}
    


base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc==1.11.0",
        "opencv-python==4.11.0.86",
        "shortuuid==1.0.13",
    )
)

this_directory = Path(__file__).parent.resolve()

server_image = base_image.add_local_dir(
    this_directory / "frontend", remote_path="/frontend"
)

with server_image.imports():
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

@app.cls(image=server_image)
class WebcamRealtimeSTTServer(ModalWebRtcSignalingServer):
    def get_modal_peer_class(self):
        return RealtimeSTTPeer

    def initialize(self):
        
        self.web_app.mount("/static", StaticFiles(directory="/frontend"))

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)


# def get_stt_model(**kwargs):
#     import RealtimeSTT
#     _recorder_config.update(kwargs)
#     return RealtimeSTT.AudioToTextRecorder(**_recorder_config)

def get_subtitle_track(track, transcription_queue):
    from aiortc import MediaStreamTrack
    import json
    import cv2
    from av import VideoFrame

    class VideoSubtitleTrack(MediaStreamTrack):
        """
        A video stream track that transforms frames from an another track.
        """

        kind = "video"

        def __init__(self, track, transcription_queue):
            super().__init__()  # don't forget this!
            self.track = track
            self.transcription_queue = transcription_queue
            self.current_transcript = ""
            async def update_transcript():
                while True:
                    try:
                        transcript_msg = await self.transcription_queue.async_get(blocking=False)
                        transcript_msg = json.loads(transcript_msg)
                        new_transcript = transcript_msg["text"]
                        if new_transcript:
                            self.current_transcript = new_transcript
                        # if transcript_msg["type"] == "final_transcript":
                        #     self.frame_count = 0
                    except Exception as e:
                        pass
                    await asyncio.sleep(0.01)
            self.update_transcript_task = asyncio.create_task(update_transcript())

        async def recv(self):
            frame = await self.track.recv()
            transcript = self.current_transcript
            
            # add subtitle to frame
            if transcript:
                import textwrap

                img = frame.to_ndarray(format="bgr24")
                h, w, _ = img.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text = transcript

                # Estimate max width for text (80% of frame width)
                max_text_width = int(w * 0.8)

                # Helper to wrap text based on pixel width
                def wrap_text(text, font, font_scale, font_thickness, max_width):
                    words = text.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        (test_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
                        if test_width <= max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    return lines

                lines = wrap_text(text, font, font_scale, font_thickness, max_text_width)

                # Calculate total height of all lines
                line_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness) for line in lines]
                text_heights = [size[0][1] for size in line_sizes]
                baselines = [size[1] for size in line_sizes]
                total_text_height = sum(text_heights) + sum(baselines) + (len(lines) - 1) * 10  # 10px between lines

                # Position: centered horizontally, 40px from bottom
                y_start = h - 40 - total_text_height + text_heights[0]
                # Find widest line for background rectangle
                text_widths = [size[0][0] for size in line_sizes]
                max_line_width = max(text_widths) if text_widths else 0

                # Draw filled rectangle as background
                rect_x1 = (w - max_line_width) // 2 - 20
                rect_y1 = y_start - text_heights[0] - 20
                rect_x2 = (w + max_line_width) // 2 + 20
                rect_y2 = h - 40 + baselines[-1] + 20
                cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

                # Draw each line of text (white), centered
                y = y_start
                for i, line in enumerate(lines):
                    (line_width, line_height), baseline = line_sizes[i]
                    x = (w - line_width) // 2
                    cv2.putText(img, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
                    y += line_height + baseline + 10  # 10px between lines
                new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
            else:
                new_frame = frame
            return new_frame
        
    return VideoSubtitleTrack(track, transcription_queue)


def get_audio_relay(queue):
    import asyncio
    import av
    from aiortc import MediaStreamTrack
    from typing import Optional


    class WebRtcAudioRelay:
        """
        A media sink that sends audio to transcription queue and discards media.
        """

        def __init__(self, queue) -> None:

            self.__tracks: dict[MediaStreamTrack, Optional[asyncio.Future]] = {}
            self.queue = queue
            self.resampler = av.audio.resampler.AudioResampler(
                format="s16",  # Target sample format (e.g., signed 16-bit integer)
                layout="mono",  # Target channel layout (e.g., stereo)
                rate=16000,  # Target sample rate (e.g., 44.1 kHz)
            )

        def add_track(self, track: MediaStreamTrack) -> None:
            """
            Add a track whose media should be discarded.

            :param track: A :class:`aiortc.MediaStreamTrack`.
            """
            if track not in self.__tracks and track.kind == "audio":
                print(f"Adding track: {track}")
                self.__tracks[track] = None

        async def start(self) -> None:
            """
            Start discarding media.
            """
            async def relay_audio(track, queue, resampler):
                while True:
                    try:
                        frame = await track.recv()

                        audioarray = frame.to_ndarray()

                        # resample array to 16000hz
                        frame = resampler.resample(frame)[0]
                        audioarray = frame.to_ndarray()
                        audiobytes = audioarray.tobytes()

                        await queue.async_put(audiobytes, blocking=False)

                    except Exception as e:
                        print(f"Audio Transcriber Error: {e}")
                        return
                    
            print("Starting tracks")
            for track, task in self.__tracks.items():
                if task is None:
                    print(f"Starting track: {track}")
                    self.__tracks[track] = asyncio.create_task(relay_audio(track, self.queue, self.resampler))

        async def stop(self) -> None:
            """
            Stop discarding media.
            """
            for task in self.__tracks.values():
                if task is not None:
                    task.cancel()
            self.__tracks = {}
        
    return WebRtcAudioRelay(queue)

def feed_audio_thread(recorder, queue, stop_event):
    """Thread function to read audio data from queue and feed it to the recorder."""
    from aiologic import QueueEmpty
    import time

    try:
        print(f"Running audio feed thread")
        while not stop_event.is_set():
            # Read a chunk of audio data from the queue
            data = None
            try:
                if queue:
                    data = queue.green_get(timeout=1.0)
                else:
                    time.sleep(0.010)
            except QueueEmpty:
                continue
            except Exception as e:
                print(f"Error in feed_audio_thread: {e}")
                continue
            # Feed the audio data to the recorder
            if data:
                recorder.feed_audio(data)   
    except Exception as e:
        print(f"\nfeed_audio_thread encountered an error: {e}")
    finally:
        print("\nFinished feeding audio.")

def recorder_transcription_thread(recorder, queues, stop_event):
    """Thread function to handle transcription and process the text."""
    import time
    import json

    transcribed_text = []
    
    def process_text(full_sentence):
        """Callback function to process the transcribed text."""
        if full_sentence and full_sentence.strip():  # Only add non-empty sentences
            transcribed_text.append(full_sentence)
            print("\nTranscribed text:", full_sentence)
            try:
                msg = {
                    "type": "final_transcript",
                    "text": full_sentence
                }
                for queue in queues:
                    queue.green_put(json.dumps(msg), blocking=False) 
            except Exception as e:
                print(f"Error in process_text while sending to sync queue: {type(e)}: {e}")
                pass

    def text_detected(text):

        if text:
            try:
                msg = {
                    "type": "realtime_transcript",
                    "text": text
                }
                for queue in queues:
                    queue.green_put(json.dumps(msg), blocking=False)

            except Exception as e:
                print(f"Error in text_detected while sending to sync queue: {type(e)}: {e}")

    recorder.on_realtime_transcription_stabilized = text_detected
    
    try:
        print("Starting transcription thread")
        # Process transcriptions until the file is done and no more text is coming
        while True:
            # Get transcribed text and process it using the callback
            recorder.text(process_text)
            if stop_event.is_set():
                break
            time.sleep(0.01)  # Small sleep to prevent CPU overuse
        
    except Exception as e:
        print(f"\ntranscription_thread encountered an error: {e}")
    finally:
        print("\nTranscription thread exiting.")

async def _send_transcription_to_client(queue, data_channel):
    print("Send transcription task started (in function)")
    from aiologic import QueueEmpty
    while True:
        try:
            text = await queue.async_get()
        except QueueEmpty:
            continue
        except Exception as e:
            print(f"Error in _send_transcription_to_client while getting from queue: {type(e)}: {e}")
            continue
        
        try:
            if data_channel:
                print(f"Sending transcription to data channel: {text}")
                data_channel.send(bytes(text, 'utf-8'))
            else:
                continue
        except Exception as e:
            print(f"Error in _send_transcription_to_client while sending to data channel: {type(e)}: {e}")