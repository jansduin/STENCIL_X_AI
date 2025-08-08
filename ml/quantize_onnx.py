import glob, cv2, numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

class Reader(CalibrationDataReader):
    def __init__(self, folder='dataset/originals'):
        self.files = iter(glob.glob(folder+'/*'))
    def get_next(self):
        try:
            p = next(self.files)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512,512)).astype('float32')/255.0
            img = img[None,None,:,:]
            return {'input': img}
        except StopIteration:
            return None

def main(inp='stencil_model_fp32.onnx', out='stencil_model_int8.onnx'):
    quantize_static(inp, out, Reader(),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True)
    print('Quantized', out)

if __name__=='__main__':
    main()
