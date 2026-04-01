# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import sys
import tempfile
from concurrent import futures

import cosyvoice_pb2
import cosyvoice_pb2_grpc

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import grpc
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../..".format(ROOT_DIR))
sys.path.append("{}/../../../third_party/Matcha-TTS".format(ROOT_DIR))

from fastcosyvoice import FastCosyVoice3

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        self.cosyvoice = FastCosyVoice3(
            model_dir="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            fp16=True,
            load_trt=True,       # TensorRT for Flow decoder (~2.5x speedup)
            load_trt_llm=True,    # TensorRT-LLM for LLM (~3x speedup)
            trt_llm_dtype="bfloat16",
            trt_llm_kv_cache_tokens=8192,
            llm_model_name="llm.pt",
            trt_llm_max_batch_size=8,
            trt_concurrent=1,
            trt_llm_concurrent_runners=4,
            temperature=0.8,
            top_k=25,
        )
        self.spk_path = None
        logging.info("grpc service initialized")

    def Inference(self, request, context):
        if request.HasField("sft_request"):
            logging.info("get sft inference request")
            model_output = self.cosyvoice.inference_sft(request.sft_request.tts_text, request.sft_request.spk_id)
        elif request.HasField("zero_shot_request"):
            logging.info("get zero_shot inference request")
            if self.spk_path is None:
                # Caching
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(request.zero_shot_request.prompt_audio)
                tmp.close()
                self.spk_path = tmp.name
                self.cosyvoice.add_zero_shot_spk(
                    f"You are a helpful assistant.<|endofprompt|>{request.zero_shot_request.prompt_text}",
                    self.spk_path,
                    "spk1",
                )
            model_output = self.cosyvoice.inference_zero_shot_stream(
                request.zero_shot_request.tts_text,
                request.zero_shot_request.prompt_text,
                self.spk_path,
                "spk1",
            )
        elif request.HasField("cross_lingual_request"):
            logging.info("get cross_lingual inference request")
            prompt_speech_16k = torch.from_numpy(
                np.array(np.frombuffer(request.cross_lingual_request.prompt_audio, dtype=np.int16))
            ).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = self.cosyvoice.inference_cross_lingual(
                request.cross_lingual_request.tts_text, prompt_speech_16k
            )
        else:
            logging.info("get instruct inference request")
            model_output = self.cosyvoice.inference_instruct(
                request.instruct_request.tts_text,
                request.instruct_request.spk_id,
                request.instruct_request.instruct_text,
            )

        logging.info("send inference response")
        for i in model_output:
            response = cosyvoice_pb2.Response()
            response.tts_audio = i
            yield response


def main():
    grpcServer = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc
    )
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)
    grpcServer.add_insecure_port("0.0.0.0:{}".format(args.port))
    grpcServer.start()
    logging.info("server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--max_conc", type=int, default=64)
    args = parser.parse_args()
    main()
