import { Link } from 'expo-router';
import { StyleSheet, View, Text } from 'react-native';

export default function Home() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>StencilX Local</Text>
      <Text style={styles.sub}>Generador de Stencil 100% local (Web + Móvil)</Text>
      <Link style={styles.btn} href="/upload">Empezar</Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex:1, alignItems:'center', justifyContent:'center', padding:24, gap:12 },
  title: { fontSize:28, fontWeight:'700' },
  sub: { fontSize:14, opacity:0.7, textAlign:'center' },
  btn: { backgroundColor:'#111', color:'#fff', paddingVertical:12, paddingHorizontal:16, borderRadius:8 }
});
