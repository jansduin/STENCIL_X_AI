import { useLocalSearchParams } from 'expo-router';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Platform, View, Text, StyleSheet } from 'react-native';
import { CanvasProcessor } from '../src/lib/canvas';
import { xdogPipeline } from '../src/lib/pipeline';
import { inferWeb, loadWebModel } from '../src/ai/onnx-web';
import { savePngWebOrRN } from '../src/lib/save';

export default function Preview() {
  const { uri } = useLocalSearchParams<{ uri:string }>();
  const [busy, setBusy] = useState(false);
  const [dataUrl, setDataUrl] = useState<string | null>(null);
  const [cfg, setCfg] = useState({sigma:1.0, k:1.6, phi:10.0, eps:-0.015, block:41, C:4, clean:60, thickness:1});
  const [useAI, setUseAI] = useState(false);
  const [aiReady, setAiReady] = useState(false);
  const canvasRef = useRef<any>(null);

  async function process() {
    if (!uri) return;
    setBusy(true);
    try {
      const cv = await CanvasProcessor.fromImageUri(String(uri), 512);
      const g = cv.getGrayFloat();
      let url: string;
      if (useAI && Platform.OS === 'web'){
        try {
          // salida IA: tensor float32 [1,1,H,W] en 0..1
          const y = await inferWeb(g, cv.w, cv.h);
          // binarizar suave: umbral adaptativo simple
          const bin = new Uint8ClampedArray(cv.w*cv.h);
          for (let i=0;i<bin.length;i++) bin[i] = y[i] > 0.5 ? 0 : 255; // líneas negras
          url = cv.writeMonochromeToCanvas(bin);
          setAiReady(true);
        } catch (e){
          console.warn('AI inference failed or model missing, fallback xDoG', e);
          setAiReady(false);
          const out = xdogPipeline(g, cv.w, cv.h, cfg);
          url = cv.writeMonochromeToCanvas(out);
        }
      } else {
        const out = xdogPipeline(g, cv.w, cv.h, cfg);
        url = cv.writeMonochromeToCanvas(out);
      }
      setDataUrl(url);
    } finally {
      setBusy(false);
    }
  }

  useEffect(()=>{ process(); }, [uri, cfg, useAI]);
  useEffect(()=>{ if (Platform.OS==='web') loadWebModel().then(()=>setAiReady(true)).catch(()=>setAiReady(false)); },[]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>2) Vista previa</Text>
      <Text style={styles.tip}>xDoG rápido en local. Opcional: "Final PRO (IA)" con ONNX en tu navegador.</Text>
      <View style={{height:8}} />
      <View style={styles.row}>
        <Text>Detalle</Text>
        <Slider val={cfg.sigma} set={(v)=>setCfg({...cfg, sigma:v})} min={0.6} max={2.0} step={0.1} />
        <Text>Grosor</Text>
        <Slider val={cfg.thickness} set={(v)=>setCfg({...cfg, thickness:Math.round(v)})} min={0} max={3} step={1} />
        <Text>Limpieza</Text>
        <Slider val={cfg.clean} set={(v)=>setCfg({...cfg, clean:Math.round(v)})} min={0} max={200} step={10} />
      </View>

      <View style={{marginTop:16}}>
        {busy && <Text>Procesando…</Text>}
        {dataUrl && <img src={dataUrl} style={{width:512, height:512, background:'#fff', borderRadius:8}} />}
      </View>

      {Platform.OS==='web' && (
        <Text style={[styles.btn,{backgroundColor: useAI?'#700':'#0a7'}]}
              onPress={()=> setUseAI(v=>!v)}>
          {useAI? 'Desactivar Final PRO (IA)' : 'Final PRO (IA)'} {aiReady? '' : '(modelo no encontrado)'}
        </Text>
      )}

      <Text style={styles.btn}
            onPress={async()=> dataUrl && await savePngWebOrRN(dataUrl, 'stencil.png')}>
        Descargar PNG
      </Text>
    </View>
  );
}

function Slider({val,set,min,max,step}:{val:number,set:(v:number)=>void,min:number,max:number,step:number}){
  // Web: input range; RN: fallback simple
  if (Platform.OS === 'web') {
    return <input type="range" min={min} max={max} step={step} defaultValue={val}
      onChange={(e:any)=>set(parseFloat(e.target.value))} style={{width:220}}/>;
  }
  return <Text onPress={()=>set(Math.min(max, val+step))} style={{padding:8,borderWidth:1}}> {val.toFixed(2)} (+) </Text>
}

const styles = StyleSheet.create({
  container: { flex:1, alignItems:'center', padding:24 },
  title: { fontSize:20, fontWeight:'700', marginTop:8 },
  tip: { textAlign:'center', opacity:0.7 },
  row: { gap:12, width:'100%', maxWidth:720, alignItems:'center', justifyContent:'center', marginTop:8, flexWrap:'wrap', flexDirection:'row' },
  btn: { marginTop:16, backgroundColor:'#111', color:'#fff', paddingVertical:12, paddingHorizontal:16, borderRadius:8 }
});
