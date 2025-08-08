import { useState } from 'react';
import { View, Text, Image, StyleSheet, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Link, useRouter } from 'expo-router';

export default function Upload() {
  const [uri, setUri] = useState<string | null>(null);
  const router = useRouter();

  async function pick() {
    const r = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: false, base64: false, quality: 1
    });
    if (!r.canceled && r.assets.length) {
      const a = r.assets[0];
      setUri(a.uri);
    }
  }
  return (
    <View style={styles.container}>
      <Text style={styles.title}>1) Cargar imagen</Text>
      <Text style={styles.tip}>Se procesará 100% en tu dispositivo. Recomendado 1024-2048px, fondo claro.</Text>
      <View style={{height:16}} />
      <Text onPress={pick} style={styles.btn}>Elegir desde galería</Text>
      {uri && <>
        <Image source={{uri}} style={{width:260, height:260, resizeMode:'contain', marginTop:16, borderRadius:8}} />
        <Link href={{pathname:'/preview', params:{ uri }}} style={[styles.btn,{backgroundColor:'#0a7'}]}>Continuar →</Link>
      </>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex:1, alignItems:'center', padding:24 },
  title: { fontSize:20, fontWeight:'700', marginTop:16 },
  tip: { textAlign:'center', opacity:0.7, marginTop:8 },
  btn: { marginTop:12, backgroundColor:'#111', color:'#fff', paddingVertical:12, paddingHorizontal:16, borderRadius:8 }
});
