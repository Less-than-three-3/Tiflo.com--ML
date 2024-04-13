from concurrent.futures import ThreadPoolExecutor
import logging
import grpc 
import warnings
import uuid
from test_pb2_grpc import AIServiceServicer, add_AIServiceServicer_to_server
warnings.filterwarnings(action = 'ignore')
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import test_pb2
import numpy as np

model = VitsModel.from_pretrained("joefox/tts_vits_ru_hf")
tokenizer = AutoTokenizer.from_pretrained("joefox/tts_vits_ru_hf")

text = "На снимке изображены две кошки"
text = text.lower()
inputs = tokenizer(text, return_tensors="pt")
inputs['speaker_id'] = 1


class ModelInit:
    def __init__(self, model_name = 't5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.model = AutoModelWithLMHead.from_pretrained('t5-base')

class AIServiceService(AIServiceServicer):
    def __init__(self):
        pass
    def VoiceTheText(self, request):
        logging.info(f"Full Text " + str(self.text))
        text = str(self.text)
        print(text)
        ###################
        
        inputs = tokenizer(text, return_tensors="pt")
        inputs['speaker_id'] = 1
        with torch.no_grad():
            output = model(**inputs).waveform

        filename = str(uuid.uuid4()) + ".wav"

        scipy.io.wavfile.write(f"/data/{filename}", rate=model.config.sampling_rate, data=output[0].cpu().numpy())

        ###################
        print("saved")
        output_int = np.int16(output[0].cpu().numpy() * 32767)  # От -32767 до 32767

# Преобразовать в байты
        audio = test_pb2.Audio()
        audio.audio = filename
           
        return audio
    
print(123)
if __name__ == "__main__":

    logging.basicConfig(
        level = logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    server = grpc.server(ThreadPoolExecutor())
    add_AIServiceServicer_to_server(AIServiceService, server)
    server.add_insecure_port('0.0.0.0:8080')
    server.start()
    logging.info(f'Server Ready')
    server.wait_for_termination()