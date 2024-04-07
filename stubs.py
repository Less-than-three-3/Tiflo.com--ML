class LLavaModelStub:
    def generate(*args, **kwargs):
        array = [[0,0,0]] * 2
        return array


class TensorStub:
    def to(*args, **kwargs):
        return {'a': 'b'}

class LLavaProcessorStub:
    def __call__(*args, **kwargs):
        return TensorStub()

    def decode(*args, **kwargs):
        prompt_len = 45
        text = "0" * prompt_len + "The simplest text for testing"
        return text
