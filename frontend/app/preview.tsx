import { useLocalSearchParams } from 'expo-router';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Platform, View, Text, StyleSheet } from 'react-native';
import { CanvasProcessor } from '../src/lib/canvas';
import { xdogPipeline } from '../src/lib/pipeline';
import { savePngWebOrRN } from '../src/lib/save';

export default function Preview() {
  const { uri } = useLocalSearchParams<{ uri:string }>();
  const [busy, setBusy] = useState(false);
  const [dataUrl, setDataUrl] = useState<string | null>(null);
  const [cfg, setCfg] = useState({sigma:1.0, k:1.6, phi:10.0, eps:-0.015, block:41, C:4, clean:60, thickness:1});
  const canvasRef = useRef<any>(null);

  async function process() {
    if (!uri) return;
    setBusy(true);
    try {
      const cv = await CanvasProcessor.fromImageUri(String(uri), 512);
      const g = cv.getGrayFloat();
      const out = xdogPipeline(g, cv.w, cv.h, cfg);
      const url = cv.writeMonochromeToCanvas(out);
      setDataUrl(url);
    } finally {
      setBusy(false);
    }
  }

  useEffect(()=>{ process(); }, [uri, cfg]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>2) Vista previa (modo rápido sin IA)</Text>
      <Text style={styles.tip}>Ajusta detalle/grosor/limpieza. Para “final PRO” podrás activar IA (ONNX) más adelante.</Text>
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
