import katex from 'katex';

/**
 * `$..$` 와 `$$..$$` 를 빌드 시점에 KaTeX 로 렌더합니다.
 * 브라우저에는 KaTeX 자바스크립트가 실리지 않고 완성된 HTML 만 나갑니다.
 *
 * 왜 display 와 inline 을 다른 단계에서 처리하는가:
 *
 *  - **display (`$$..$$`)** 는 mdast 단계에서 처리합니다.
 *    Sätteri 는 display 수식을 `<pre><code class="language-math math-display">`
 *    로 내보내는데, 이 code 노드에는 `data.lang` 이 없습니다. Astro 의 Shiki
 *    하이라이터는 `lang ?? 'plaintext'` 로 읽기 때문에 `excludeLangs: ['math']`
 *    로도 걸러지지 않고, 사용자 hast 플러그인보다 **먼저** 돌면서 수식을
 *    코드블록으로 만들어 버립니다. mdast 는 그 앞 단계라 안전합니다.
 *
 *  - **inline (`$..$`)** 은 hast 단계에서 처리합니다.
 *    mdast 의 `raw` 는 마크다운으로 재파싱되기 때문에, 문단 중간에 넣으면
 *    `<p>` 가 새로 열리면서 문장이 끊깁니다. hast 의 `raw` 는 재파싱 없이
 *    그대로 들어갑니다. 하이라이터는 `<pre>` 만 보므로 inline 은 안 건드립니다.
 *
 * rehype-katex 를 못 쓰는 이유: Sätteri 플러그인은 unified 계약이 아니라
 * 자체 visitor 계약을 씁니다.
 */

const RENDER_OPTIONS = {
  throwOnError: false,
  strict: false as const,
  // MathML 을 함께 넣어야 스크린 리더가 수식을 읽습니다.
  output: 'htmlAndMathml' as const,
};

type Reporter = {
  report(opts: {
    message: string;
    node?: unknown;
    severity?: 'error' | 'warning' | 'info';
  }): void;
};

function renderTex(
  tex: string,
  displayMode: boolean,
  ctx: Reporter,
  node: unknown,
): string | undefined {
  if (!tex.trim()) return undefined;
  try {
    return katex.renderToString(tex, { ...RENDER_OPTIONS, displayMode });
  } catch (error) {
    ctx.report({
      message: `KaTeX 렌더 실패: ${(error as Error).message}`,
      node,
      severity: 'warning',
    });
    return undefined;
  }
}

/** display 수식 (`$$..$$`) 담당. */
export const katexDisplayPlugin = {
  name: 'katex-display',
  math(node: { value?: string }, ctx: Reporter) {
    const html = renderTex(node.value ?? '', true, ctx, node);
    // mdxExpressions: false — KaTeX 가 내보내는 중괄호를 MDX 표현식으로
    // 해석하지 않도록 합니다.
    return html ? { raw: html, mdxExpressions: false } : undefined;
  },
};

/** inline 수식 (`$..$`) 담당. */
export const katexInlinePlugin = {
  name: 'katex-inline',
  element: {
    filter: ['code'],
    visit(node: any, ctx: any) {
      const raw = node.properties?.className;
      const classes = Array.isArray(raw) ? raw.map(String) : [];
      if (!classes.includes('math-inline')) return;

      const html = renderTex(ctx.textContent(node), false, ctx, node);
      if (html) ctx.replaceNode(node, { type: 'raw', value: html });
    },
  },
};
