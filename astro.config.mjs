// @ts-check
import { defineConfig, fontProviders } from 'astro/config';
import { satteri } from '@astrojs/markdown-satteri';
import sitemap from '@astrojs/sitemap';
import { SITE } from './src/consts.js';
import { katexDisplayPlugin, katexInlinePlugin } from './src/lib/katex.js';

export default defineConfig({
  site: SITE.url,
  integrations: [sitemap()],
  // 빌드 시 폰트를 내려받아 같이 배포합니다. 방문자 브라우저가 Google에
  // 요청하지 않고, 폰트가 늦게 떠서 글자가 튀는 현상도 없습니다.
  fonts: [
    {
      provider: fontProviders.google(),
      name: 'Hahmlet',
      cssVariable: '--font-display',
      weights: [400, 600, 800],
      subsets: ['latin', 'korean'],
      fallbacks: ['Georgia', 'serif'],
    },
    {
      provider: fontProviders.google(),
      name: 'IBM Plex Sans KR',
      cssVariable: '--font-body',
      weights: [400, 500, 600],
      subsets: ['latin', 'korean'],
      fallbacks: ['system-ui', 'sans-serif'],
    },
    {
      provider: fontProviders.google(),
      name: 'JetBrains Mono',
      cssVariable: '--font-mono',
      weights: [400, 500],
      subsets: ['latin'],
      fallbacks: ['ui-monospace', 'monospace'],
    },
  ],
  markdown: {
    // Sätteri 가 $..$ / $$..$$ 를 math 노드로 파싱하고, 두 KaTeX 플러그인이
    // 빌드 시점에 HTML 로 렌더합니다. 브라우저로는 KaTeX JS 가 안 나갑니다.
    // display 와 inline 을 왜 다른 단계에서 처리하는지는 src/lib/katex.ts 참고.
    processor: satteri({
      features: { math: true },
      mdastPlugins: [katexDisplayPlugin],
      hastPlugins: [katexInlinePlugin],
    }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      wrap: false,
    },
  },
});
