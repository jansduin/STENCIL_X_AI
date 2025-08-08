// xDoG-based stencil pipeline in pure TS (Web).
// Includes: Gaussian blur, DoG, xDoG nonlinearity, adaptive threshold, morphology, cleanup

function gaussianKernel1D(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const size = radius*2 + 1;
  const k = new Float32Array(size);
  let sum = 0;
  for (let i=0;i<size;i++){
    const x = i - radius;
    const v = Math.exp(-(x*x)/(2*sigma*sigma));
    k[i] = v; sum += v;
  }
  for (let i=0;i<size;i++) k[i] /= sum;
  return k;
}

function conv1D(src: Float32Array, w:number, h:number, kernel: Float32Array, horiz: boolean): Float32Array {
  const r = (kernel.length-1)/2|0;
  const dst = new Float32Array(w*h);
  if (horiz){
    for (let y=0;y<h;y++){
      for (let x=0;x<w;x++){
        let acc=0;
        for (let k=-r;k<=r;k++){
          const xx = Math.min(w-1, Math.max(0, x+k));
          acc += src[y*w+xx] * kernel[k+r];
        }
        dst[y*w+x] = acc;
      }
    }
  } else {
    for (let y=0;y<h;y++){
      for (let x=0;x<w;x++){
        let acc=0;
        for (let k=-r;k<=r;k++){
          const yy = Math.min(h-1, Math.max(0, y+k));
          acc += src[yy*w+x] * kernel[k+r];
        }
        dst[y*w+x] = acc;
      }
    }
  }
  return dst;
}

function gaussianBlur(src: Float32Array, w:number, h:number, sigma:number): Float32Array {
  const k = gaussianKernel1D(sigma);
  const t = conv1D(src, w, h, k, true);
  return conv1D(t, w, h, k, false);
}

function adaptiveThresholdMean(src: Float32Array, w:number, h:number, block=41, C=4): Uint8ClampedArray {
  const r = (block-1)/2|0;
  // integral image for fast mean
  const integral = new Float64Array((w+1)*(h+1));
  for (let y=1;y<=h;y++){
    let rowsum = 0;
    for (let x=1;x<=w;x++){
      rowsum += src[(y-1)*w + (x-1)];
      integral[y*(w+1)+x] = integral[(y-1)*(w+1)+x] + rowsum;
    }
  }
  const out = new Uint8ClampedArray(w*h);
  for (let y=0;y<h;y++){
    const y0=Math.max(0,y-r), y1=Math.min(h-1,y+r);
    for (let x=0;x<w;x++){
      const x0=Math.max(0,x-r), x1=Math.min(w-1,x+r);
      const A = integral[y0*(w+1)+x0];
      const B = integral[y0*(w+1)+x1+1];
      const Cc= integral[(y1+1)*(w+1)+x0];
      const D = integral[(y1+1)*(w+1)+x1+1];
      const area = (x1-x0+1)*(y1-y0+1);
      const mean = (D - B - Cc + A)/area;
      const thr = mean - (C/255.0);
      out[y*w+x] = src[y*w+x] > thr ? 255 : 0;
    }
  }
  return out;
}

function morph(bin: Uint8ClampedArray, w:number, h:number, thickness=1): Uint8ClampedArray {
  // simple dilate/erode sequence based on thickness
  let out = bin.slice(0);
  const iter = Math.max(0, thickness|0);
  if (iter === 0) return out;
  // dilate then erode to thicken (closing-like)
  for (let t=0;t<iter;t++) out = dilate(out, w, h);
  for (let t=0;t<iter;t++) out = erode(out, w, h);
  return out;
}

function dilate(bin: Uint8ClampedArray, w:number, h:number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w*h);
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      let m=0;
      for (let j=-1;j<=1;j++){
        for (let i=-1;i<=1;i++){
          const yy=Math.min(h-1,Math.max(0,y+j));
          const xx=Math.min(w-1,Math.max(0,x+i));
          if (bin[yy*w+xx]===255){ m=255; break; }
        }
        if (m===255) break;
      }
      out[y*w+x]=m;
    }
  }
  return out;
}

function erode(bin: Uint8ClampedArray, w:number, h:number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w*h);
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      let m=255;
      for (let j=-1;j<=1;j++){
        for (let i=-1;i<=1;i++){
          const yy=Math.min(h-1,Math.max(0,y+j));
          const xx=Math.min(w-1,Math.max(0,x+i));
          if (bin[yy*w+xx]===0){ m=0; break; }
        }
        if (m===0) break;
      }
      out[y*w+x]=m;
    }
  }
  return out;
}

export function xdogPipeline(gray: Float32Array, w:number, h:number, cfg: {sigma:number,k:number,phi:number,eps:number,block:number,C:number,clean:number, thickness:number}){
  const {sigma,k,phi,eps,block,C,clean, thickness} = cfg;
  // Pre: mild blur for denoise preserve edges
  const g1 = gaussianBlur(gray, w, h, sigma);
  const g2 = gaussianBlur(gray, w, h, sigma*k);
  const dog = new Float32Array(w*h);
  for (let i=0;i<dog.length;i++) dog[i] = g1[i] - g2[i];

  // xDoG nonlinearity
  const X = new Float32Array(w*h);
  for (let i=0;i<X.length;i++){
    X[i] = 1.0 + 20.0 * Math.tanh(phi*(dog[i] - eps));
    X[i] = Math.min(1, Math.max(0, X[i]));
    X[i] = 1.0 - X[i]; // invert for black lines
  }

  // Adaptive threshold
  let bin = adaptiveThresholdMean(X, w, h, block, C);

  // Morph thickness
  bin = morph(bin, w, h, thickness);

  // Cleanup: remove tiny components (simple pass)
  if (clean > 0){
    const mark = new Uint8Array(w*h);
    function dfs(sx:number, sy:number, id:number): number{
      const stack = [[sx,sy]]; let area = 0;
      while(stack.length){
        const [x,y]=stack.pop()!; const idx=y*w+x;
        if (mark[idx]||bin[idx]===0) continue;
        mark[idx]=1; area++;
        for (let j=-1;j<=1;j++) for (let i=-1;i<=1;i++){
          const xx=x+i, yy=y+j;
          if (xx>=0&&yy>=0&&xx<w&&yy<h) stack.push([xx,yy]);
        }
      }
      return area;
    }
    const keep = new Uint8Array(w*h);
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){
      const idx=y*w+x;
      if (bin[idx]===255 && !mark[idx]){
        const area = dfs(x,y,1);
        if (area >= clean){
          // flood again to keep
          const st=[[x,y]];
          while(st.length){
            const [xx,yy]=st.pop()!; const i2=yy*w+xx;
            if (keep[i2]||bin[i2]===0) continue;
            keep[i2]=255;
            for (let j=-1;j<=1;j++) for (let i=-1;i<=1;i++){
              const nx=xx+i, ny=yy+j;
              if (nx>=0&&ny>=0&&nx<w&&ny<h) st.push([nx,ny]);
            }
          }
        }
      }
    }
    bin = keep as any;
  }
  return bin;
}
