import { createSession } from './ui/session.js';
const session = createSession((message, transfer = []) => postMessage(message, transfer));
self.onmessage = ({ data }) => { session.handle(data); };
self.addEventListener('error', (event) => postMessage({ type: 'error', message: event.message }));
self.addEventListener('unhandledrejection', (event) => postMessage({ type: 'error', message: String(event.reason) }));
