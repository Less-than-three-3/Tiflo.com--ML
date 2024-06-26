import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import yaml
import hydra
from omegaconf import DictConfig

from PIL import Image, UnidentifiedImageError
import torch
import translators as ts
import grpc

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration

from protos.img2seq_pb2_grpc import (
    ImageCaptioningServicer,
    add_ImageCaptioningServicer_to_server,
)
from protos.img2seq_pb2 import Text

from stubs import LLavaModelStub, LLavaProcessorStub

warnings.filterwarnings(action="ignore")

LOGGER_NAME = "IMAGE2TEXT_MODEL_SERVER"
MODEL_LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_BLIP_ID = "Salesforce/blip-image-captioning-large"

logger = logging.getLogger(LOGGER_NAME)



class ImageCaptionModel:
    """
    Модель генерации текста к картинке

    Attributes
    ----------
    prompt: str
        Промпт для модели
    device: str
        'cpu' или 'cuda'
    torch_dtype: torch.*.FloatTensor
        Точность для чисел с плавающей запятой.
        Принимает значения torch.float16 (cuda) или torch.float32 (cpu)
    model: LlavaForConditionalGeneration | BlipForConditionalGeneration
        Модель для описания изображения
    model_type: str
        Тип модели (Llava, blip, etc)
    processor: AutoProcessor
        Преобразование в токены и обратно
    """

    model_type: str
    use_only_cuda: bool
    use_translator: bool
    translator: str

    prompt = "USER: <image>\nwhat is shown in the image?\nASSISTANT:"

    def __init__(self, model_type, use_only_cuda, use_translator, translator) -> None:
        """
        Parameters
        ----------
        model_type: str
            Тип модели (Llava, Blip или Stub (заглушка))
        use_only_cuda: bool
            Если True, то в случае отсуствия cuda вызывается исключение
        use_translator: bool
            Если True, то сгенерированный английский текст переводится
            на русский язык
        translator: str
            Используемый переводчик ('google', 'yandex' ...)

        Raises
        ------
        RuntimeError
            Вызывается в случае недоступности cuda
            и если use_only_cuda = True
        ValueError
            Вызывается, если model_type принимает
            неизвестное значение
        """
        if torch.cuda.is_available():
            logger.info("Use cuda")
            self.device = "cuda"
            self.torch_dtype = torch.float16
        else:
            logger.warning("Cuda is not available")
            if use_only_cuda:
                raise RuntimeError("Only cuda available")

            self.device = "cpu"
            self.torch_dtype = torch.float32

        self.model_type = model_type.upper()
        match self.model_type:
            case "LLAVA":
                self.model = (
                    LlavaForConditionalGeneration.from_pretrained(
                        MODEL_LLAVA_ID,
                        torch_dtype=self.torch_dtype,
                        low_cpu_mem_usage=True,
                    )
                    .eval()
                    .to(self.device)
                )

                self.processor = AutoProcessor.from_pretrained(MODEL_LLAVA_ID)
                logger.info("Use Llava model. Model ID: %s", MODEL_LLAVA_ID)

            case "BLIP":
                self.model = BlipForConditionalGeneration.from_pretrained(MODEL_BLIP_ID)
                self.processor = BlipProcessor.from_pretrained(MODEL_BLIP_ID)
                logger.info("Use Blip model. Model ID: %s", MODEL_BLIP_ID)

            case "STUB":
                self.model = LLavaModelStub()
                self.processor = LLavaProcessorStub()
                logger.info("Use stub model")

            case _:
                raise ValueError(f"Неизвестная модель описания изображений: {model_type}")

        self.use_translator = use_translator
        self.translator = translator

    def generate_text(self, image, request_id=None):
        """
        Генерация текста на английском по изображению

        Parameters
        ----------
        image: PIL.Image
            Изображение, которому нужны комменатрии
        request_id: str | int | None
            Идентификатор запроса

        Returns
        -------
        str
            Сгенерированный и переведённый текст
        """

        with torch.no_grad():
            start = time.time()
            try:
                match self.model_type:
                    # Каждый случай должен вернуть переменную text
                    case "BLIP":
                        inputs = self.processor(image, return_tensors="pt")
                        output = self.model.generate(**inputs)
                        text = self.processor.decode(
                            output[0], skip_special_tokens=True
                        ).capitalize()

                    case "LLAVA" | "STUB":
                        prompt_len = 45
                        inputs = self.processor(
                            self.prompt, image, return_tensors="pt"
                        ).to(self.device, self.torch_dtype)
                        output = self.model.generate(
                            **inputs, max_new_tokens=200, do_sample=False
                        )
                        output_text = self.processor.decode(
                            output[0][2:], skip_special_tokens=True
                        )
                        text = (
                            output_text[prompt_len:]
                            .replace("The image shows ", "")
                            .capitalize()
                        )

            except Exception as err:
                logger.error(
                    "[ID: %s] Непредвиденная ошибка работы модели! Сообщение об ошибке: %s",
                    request_id,
                    err,
                )
                return "[ERROR]"

            logger.debug(
                "[ID: %s] Время описания изображения: %.2f",
                request_id,
                time.time() - start,
            )

        if self.use_translator:
            text = self.translate(text, request_id)

        return text

    def translate(self, en_text, request_id=None):
        """
        Перевод текста

        Parameters
        ----------
        en_text: str
            Текст на английском языке
        request_id: str | int | None
            Идентификатор запроса

        Returns
        -------
        str
            Текст на русском языке
        """
        start_translate = time.time()
        try:
            translation = ts.translate_text(
                en_text,
                translator=self.translator,
                from_language="en",
                to_language="ru",
            )
        except ts.server.TranslatorError as err:
            logger.error(
                "[ID: %s] Ошибка при попытке перевода текста! Сообщение об ошибке: %s",
                request_id,
                err,
            )
            return en_text

        logger.debug(
            "[ID: %s] Время перевода: %.2f", request_id, time.time() - start_translate
        )
        return translation


class ImageCaptioningService(ImageCaptioningServicer):
    """
    Сервер на gRPC с моделью ImageCaptionModel.

    Attributes
    ----------
    model: ImageCaptioningModel
        Модель для генерации текста
    """

    def __init__(self, cfg):
        """
        Parameters
        ----------
        cfg: DictConfig
            Конфигурация модели
        """
        self.model = hydra.utils.instantiate(cfg.model)

    def ImageCaption(self, request, context):
        """
        Метод, вызываемый по gRPC.

        Parameters
        ----------
        request: object
            Запрос на генерацию текста.
            Содержит путь до файла image_path
        context: object
            Контекст gRPC

        Returns
        -------
        str
            Сгенерированный текст
        """
        request_id = int(time.time() * 100)
        logger.info("[ID: %s] Запрос на описание фото.", request_id)
        logger.debug("[ID: %s] Image path: %s", request_id, request.image_path)
        output_text = Text()
        try:
            image = Image.open(request.image_path)
        except FileNotFoundError:
            logger.error("Файл по пути %s не найден!", request.image_path)
            output_text.text = "[ERROR]"
            return output_text
        except UnidentifiedImageError as err:
            logger.error(
                "Ошибка открытия файла по пути '%s'! Сообщение об ошибке: %s",
                request.image_path,
                err,
            )
            output_text.text = "[ERROR]"
            return output_text

        generated_text = self.model.generate_text(image, request_id)
        output_text.text = generated_text
        logger.info("[ID: %s] Текст отправлен.", request_id)
        return output_text


def logging_init():
    """
    Инициализация логгера из yaml файла
    """
    logging_conf_path = "./config/logging.yaml"
    with open(logging_conf_path, "r") as file:
        logging_conf = yaml.load(file, Loader=yaml.FullLoader)

    logging.config.dictConfig(logging_conf)
    logger.info("Init logger")


@hydra.main(version_base=None, config_path="config", config_name="imagecap_server")
def start_image_captioning_server(cfg: DictConfig):
    """
    Запуск сервера

    Parameters
    ----------
    cfg: DictConfig
        Конфигурация сервера (ip и порт) и модели
    """
    server = grpc.server(ThreadPoolExecutor())
    add_ImageCaptioningServicer_to_server(ImageCaptioningService(cfg), server)
    port = cfg.server.port
    ip = cfg.server.ip
    server.add_insecure_port(f"{ip}:{port}")

    server.start()
    logger.info("Server Ready")
    server.wait_for_termination()


if __name__ == "__main__":
    logging_init()
    start_image_captioning_server()
