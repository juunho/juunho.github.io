import { defineCollection, z } from 'astro:content';
import { glob, file } from 'astro/loaders';
import { load as parseYaml } from 'js-yaml';

/** 여기서 직접 쓴 글. src/content/posts/*.md */
const posts = defineCollection({
  loader: glob({ base: './src/content/posts', pattern: '**/*.md' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    /** 목록과 <meta description>에 쓰이는 한 줄 요약. */
    description: z.string().optional(),
    tags: z.array(z.string()).default([]),
    /** true 면 빌드에서 제외됩니다 (dev 서버에서는 보입니다). */
    draft: z.boolean().default(false),
  }),
});

/** 해온 일. src/content/work/*.md — 본문은 안 쓰고 프론트매터만 씁니다. */
const work = defineCollection({
  loader: glob({ base: './src/content/work', pattern: '**/*.md' }),
  schema: z.object({
    title: z.string(),
    /** 오른쪽에 작게 붙는 분류. 예: Retrieval, Training, Serving, Open Source */
    kind: z.string(),
    /**
     * 왼쪽 기간 컬럼에 그대로 출력됩니다. 예: "2025 — 현재"
     * `period: 2026` 처럼 따옴표 없이 써도 되도록 문자열로 변환합니다.
     */
    period: z.coerce.string(),
    /** 정렬 기준. 큰 값이 위로 옵니다. */
    order: z.number(),
    summary: z.string(),
    stack: z.array(z.string()).default([]),
    /** 선택. 관련 글이나 레포 링크 한 개. */
    link: z
      .object({
        href: z.string(),
        label: z.string(),
      })
      .optional(),
  }),
});

/**
 * 다른 데 쓴 글 (북마크). src/content/links.yaml
 *
 * id 를 직접 쓸 필요가 없도록 파서에서 url 로부터 만들어 넣습니다.
 * 항목 하나 추가 = 아래 4~5줄:
 *
 *   - title: "RAG 검색 품질을 어떻게 측정할 것인가"
 *     date: 2026-07-11
 *     url: https://velog.io/@juunho/rag-eval
 *     source: velog
 *     tags: [RAG, Evaluation]
 */
const links = defineCollection({
  loader: file('./src/content/links.yaml', {
    parser: (text) => {
      const raw = parseYaml(text);
      if (!Array.isArray(raw)) return [];
      return raw.map((entry, i) => {
        const item = entry as Record<string, unknown>;
        return { id: slugifyUrl(String(item.url ?? i)), ...item };
      });
    },
  }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    url: z.string().url(),
    /** 목록 오른쪽에 "↗ velog" 처럼 출력됩니다. */
    source: z.string(),
    tags: z.array(z.string()).default([]),
    /** 선택. 왜 걸어뒀는지 한 줄. */
    note: z.string().optional(),
  }),
});

function slugifyUrl(url: string): string {
  return url
    .replace(/^https?:\/\//, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase()
    .slice(0, 90);
}

export const collections = { posts, work, links };
