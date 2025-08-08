import torch, sys
from train_unet import UNetSmall
def main(pth='stencil_model_fp32.pth', onnx_out='stencil_model_fp32.onnx'):
    net = UNetSmall()
    net.load_state_dict(torch.load(pth, map_location='cpu'))
    net.eval()
    x = torch.randn(1,1,512,512)
    torch.onnx.export(net, x, onnx_out,
                      input_names=['input'], output_names=['output'],
                      opset_version=17, do_constant_folding=True)
    print('Exported to', onnx_out)
if __name__=='__main__': main(*sys.argv[1:])
