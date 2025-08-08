import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';

export async function savePngWebOrRN(dataUrl: string, filename: string){
  if (Platform.OS === 'web'){
    const a = document.createElement('a');
    a.href = dataUrl; a.download = filename; a.click();
    return;
  }
  const base64 = dataUrl.split(',')[1];
  const path = FileSystem.documentDirectory + filename;
  await FileSystem.writeAsStringAsync(path, base64, { encoding: FileSystem.EncodingType.Base64 });
  alert('Guardado en: ' + path);
}
