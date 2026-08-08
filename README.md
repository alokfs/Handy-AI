# Handy AI

Handy AI is a static HTML, CSS, and JavaScript hand-gesture whiteboard. It uses the browser camera and MediaPipe Hand Landmarker to detect gestures locally in the browser. There is no Python backend, server process, or Python dependency.

## Use it

1. Publish this folder to any static HTTPS host (Netlify, GitHub Pages, Vercel, or Cloudflare Pages), then open the deployed URL.
2. Select **Start camera** and allow camera access.
3. Use an index finger to write. Make a fist and hold briefly to clear. Show thumb + index to pause. Show an open palm and hold it briefly to solve.
4. To enable solving, enter your own Gemini API key in the page. It is kept in `sessionStorage` only for the active browser tab.

The app may also be previewed on `http://localhost` with any static file server, for example `npx serve .`. Do not open `index.html` directly from disk: browsers require an HTTPS page or `localhost` before they grant camera access.

## AI key security

This project intentionally has no backend. That means a hard-coded production API key would be visible to every visitor, so the project never stores one in its source files. Each person enters their own key for the current tab and the browser sends the drawing directly to Gemini.

For a public product with a shared API key, use a secure server-side proxy or authentication service. Do not put an API key in `script.js`, HTML, or a static-host environment variable exposed to the client.

## Project files

```
Handy-AI-main/
├── index.html
├── static/
│   ├── style.css
│   └── script.js
└── DEPLOY.md
```

The first visit downloads MediaPipe's browser runtime and hand-landmark model from its CDN. After that, hand landmark detection runs on the visitor's device.
