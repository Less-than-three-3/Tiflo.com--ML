import logging
import time
import warnings
import yaml
from concurrent.futures import ThreadPoolExecutor

import hydra
from omegaconf import OmegaConf, DictConfig

from PIL import Image
from io import BytesIO
import torch
import translators as ts
import grpc 

from transformers import AutoProcessor, LlavaForConditionalGeneration

from protos.img2seq_pb2_grpc import ImageCaptioningServicer, add_ImageCaptioningServicer_to_server
from protos.img2seq_pb2 import Text

from stubs import LLavaModelStub, LLavaProcessorStub

warnings.filterwarnings(action = 'ignore')

logger = logging.getLogger(__name__)


class ImageCaptionModel:
    model_id: str
    use_only_cuda: bool
    use_translator: bool
    translator: str

    def __init__(self, model_id, use_only_cuda, use_translator, translator) -> None:
        if torch.cuda.is_available():
            logger.info('Use cuda')
            self.device = 'cuda'
        else:
            logger.warning('Cuda is not available')
            if use_only_cuda:
                raise RuntimeError('Only cuda available')
            
            self.device = 'cpu'
        
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #                                                             model_id, 
        #                                                             torch_dtype=torch.float16, 
        #                                                             low_cpu_mem_usage=True,
        #                                                         ).eval().to(self.device)
        
        # self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.model = LLavaModelStub()
        self.processor = LLavaProcessorStub()

        self.prompt = "USER: <image>\nwhat is shown in the image?\nASSISTANT:"

    def generate_text(self, image):
        prompt_len = 45

        with torch.no_grad():
            start = time.time()
            raw_image = Image.open(image)
            inputs = self.processor(self.prompt, raw_image, return_tensors='pt').to(self.device, torch.float16)

            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            text = self.processor.decode(output[0][2:], skip_special_tokens=True)[prompt_len:]

            logger.debug(f'Время описания изображения: {time.time() - start}')

        if self.use_translator:
            text = self.translate(text)

        return text

    def translate(self, en_text):
        start_translate = time.time()
        translation = ts.translate_text(en_text, translator=self.translator, from_language='en', to_language='ru')
        logger.debug(f'Время перевода: {time.time() - start_translate}' )

        return translation


class ImageCaptioningServicer(ImageCaptioningServicer):
    def __init__(self, cfg):
        self.model = hydra.utils.instantiate(cfg.model)

    def ImageCaption(self, request):
        image = Image.open(BytesIO(self.image))
        # image.show()
        generated_text = self.model.generate_text(image)
        output_text = Text()
        output_text.text = generated_text
        return output_text


def logging_init():
    logging_conf_path = './config/logging.yaml'
    with open(logging_conf_path, 'r') as file:
        logging_conf = yaml.load(file, Loader=yaml.FullLoader)
    
    logging.config.dictConfig(logging_conf)
    global logger
    logger = logging.getLogger('img_server')
    logger.info('Init logger')


@hydra.main(version_base=None, config_path="config", config_name="imagecap_server")
def start_image_captioning_server(cfg: DictConfig):
    server = grpc.server(ThreadPoolExecutor())
    add_ImageCaptioningServicer_to_server(ImageCaptioningServicer(cfg), server)
    port = cfg.server.port
    ip = cfg.server.ip
    server.add_insecure_port(f'{ip}:{port}')

    server.start()
    logger.info(f'Server Ready')
    server.wait_for_termination()


if __name__ == '__main__':
    logging_init()
    start_image_captioning_server()

