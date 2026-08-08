# Static deployment

Handy AI is a static site. Deploy the repository root as the published directory; no build command, runtime, dependencies, or environment variables are required.

## Netlify

1. Create a new site from this repository.
2. Set the publish directory to `.` (the repository root).
3. Leave the build command empty.
4. Deploy and open the HTTPS site URL.

## Other hosts

For GitHub Pages, Vercel, Cloudflare Pages, or similar, publish these files unchanged:

```
index.html
static/style.css
static/script.js
```

Use HTTPS in production. Browsers allow camera access only in a secure context (HTTPS) or on `localhost` during local development.

Each visitor supplies their own Gemini API key in the page if they want AI solutions. No API key should be configured in the static host or committed to the repository.
