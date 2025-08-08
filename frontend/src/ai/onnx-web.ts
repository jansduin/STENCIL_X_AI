// Hook for local ONNX inference on Web (drop model in /public/models/stencil_model_int8.onnx)
import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;

export async function loadWebModel(url='/models/stencil_model_int8.onnx'){
  if (session) return session;
  session = await ort.InferenceSession.create(url, {
    executionProviders: ['webgpu','wasm']
  });
  return session;
}

export async function inferWeb(gray: Float32Array, w:number, h:number){
  const s = await loadWebModel();
  const input = new ort.Tensor('float32', gray, [1,1,h,w]);
  const out = await s.run({ input });
  return out.output.data as Float32Array;
}
