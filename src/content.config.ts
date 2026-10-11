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
    /**
     * 글의 언어. 비우면 사이트 언어(src/consts.ts 의 SITE.lang)를 따릅니다.
     * <html lang> 과 읽는 시간 계산(한글은 글자 수, 영어는 단어 수)에 쓰입니다.
     */
    lang: z.string().optional(),
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
 * 파일은 `links:` 키 아래 목록을 담습니다. 최상위가 배열이면 Git 기반 CMS
 * (/admin/) 가 파일을 편집하지 못하기 때문입니다.
 * id 는 파서에서 url 로부터 만들어 넣으므로 직접 쓸 필요가 없습니다.
 *
 *   links:
 *     - title: "RAG 검색 품질을 어떻게 측정할 것인가"
 *       date: 2026-07-11
 *       url: https://velog.io/@juunho/rag-eval
 *       source: velog
 *       tags: [RAG, Evaluation]
 */
const links = defineCollection({
  loader: file('./src/content/links.yaml', {
    parser: (text) => {
      const doc = parseYaml(text) as { links?: unknown } | null;
      const list = doc?.links;
      if (!Array.isArray(list)) return [];
      return list.map((entry, i) => {
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

/**
 * 사이트 문구. src/content/site.yaml 하나가 'site' 라는 항목 하나가 됩니다.
 * 어드민의 Site 폼이 이 파일을 편집합니다.
 */
const site = defineCollection({
  loader: file('./src/content/site.yaml', {
    parser: (text) => {
      const doc = parseYaml(text);
      if (!doc || typeof doc !== 'object' || Array.isArray(doc)) return [];
      return [{ id: 'site', ...(doc as Record<string, unknown>) }];
    },
  }),
  schema: z.object({
    headline_top: z.string(),
    headline_accent: z.string(),
    bio: z.string(),
    description: z.string(),
    social: z
      .array(z.object({ label: z.string(), href: z.string() }))
      .default([]),
  }),
});

/** 단독 페이지 본문. 지금은 About 하나. src/content/pages/*.md */
const pages = defineCollection({
  loader: glob({ base: './src/content/pages', pattern: '**/*.md' }),
  schema: z.object({
    title: z.string(),
  }),
});

export const collections = { posts, work, links, site, pages };
