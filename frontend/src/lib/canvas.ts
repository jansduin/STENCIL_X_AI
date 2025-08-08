export class CanvasProcessor {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;

  private constructor(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, w:number, h:number){
    this.canvas = canvas; this.ctx = ctx; this.w = w; this.h = h;
  }

  static async fromImageUri(uri: string, target: number = 512): Promise<CanvasProcessor> {
    if (typeof document === 'undefined') throw new Error('Canvas only available on web for now.');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = uri;
    await new Promise((res, rej)=>{ img.onload = ()=>res(null); img.onerror = rej; });
    const scale = target / Math.max(img.width, img.height);
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0, w, h);
    return new CanvasProcessor(canvas, ctx, w, h);
  }

  getGrayFloat(): Float32Array {
    const im = this.ctx.getImageData(0,0,this.w,this.h);
    const out = new Float32Array(this.w*this.h);
    const d = im.data;
    for (let i=0, j=0;i<d.length;i+=4, j++){
      out[j] = (0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2]) / 255.0;
    }
    return out;
  }

  writeMonochromeToCanvas(bin: Uint8ClampedArray): string {
    const im = this.ctx.createImageData(this.w, this.h);
    for (let i=0, j=0; j<bin.length; i+=4, j++){
      const v = bin[j];
      im.data[i] = v; im.data[i+1] = v; im.data[i+2] = v; im.data[i+3] = 255;
    }
    this.ctx.putImageData(im, 0, 0);
    return this.canvas.toDataURL('image/png');
  }
}
