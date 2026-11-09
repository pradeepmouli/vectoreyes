import { AppBridge, PostMessageTransport } from '@modelcontextprotocol/ext-apps/app-bridge';

const fixtureOrigin = 'http://127.0.0.1:62746';
const response = await fetch(`${fixtureOrigin}/response`).then((value) => value.json());
const frame = document.querySelector('iframe');
const scenario = new URLSearchParams(location.search).get('scenario');
const bridge = new AppBridge(null, { name: 'vectorEyes local preview', version: '1' }, {
  serverTools: {}, message: { text: {} }, updateModelContext: { text: {} },
});
window.previewEvents = [];
window.previewBridge = bridge;
bridge.oncalltool = async (request) => {
  window.previewEvents.push(request);
  if (scenario === 'error') return { isError: true, content: [{ type: 'text', text: 'Preview artifact is unavailable. Choose another view or ask your agent to retry.' }] };
  if (request.name !== 'vectoreyes_get_drawing_artifact') return { isError: true, content: [{ type: 'text', text: 'This preview does not run conversions.' }] };
  const { artifact, version } = request.arguments;
  return fetch(`${fixtureOrigin}/artifact?version=${encodeURIComponent(version)}&kind=${encodeURIComponent(artifact)}`).then((value) => value.json());
};
bridge.onmessage = async (message) => { window.previewEvents.push({ message }); return {}; };
bridge.onupdatemodelcontext = async () => ({});
bridge.onsizechange = ({ height }) => { if (height) frame.style.height = `${Math.ceil(height)}px`; };
bridge.oninitialized = async () => {
  bridge.setHostContext({ theme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light' });
  if (scenario === 'generating') await bridge.sendToolInput({ arguments: { image: { download_url: 'https://example.invalid/preview.png' } } });
  else if (scenario !== 'empty') await bridge.sendToolResult({ structuredContent: response, content: [] });
};
await bridge.connect(new PostMessageTransport(frame.contentWindow, frame.contentWindow));
frame.src = '/mcp-app.html';
