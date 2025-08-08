import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
export default function Layout() {
  return (
    <>
      <Stack screenOptions={{ headerTitle: 'StencilX Local', headerBackTitle: 'Back' }} />
      <StatusBar style="dark" />
    </>
  );
}
