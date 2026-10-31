import { createSession } from './ui/session.js';
const session = createSession((message, transfer = []) => postMessage(message, transfer));
self.onmessage = ({ data }) => { session.handle(data); };
