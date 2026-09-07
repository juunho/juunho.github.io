import { getCollection } from 'astro:content';

/**
 * 여기 쓴 글과 다른 데 쓴 글을 하나의 스트림으로 합칩니다.
 * 목록에서는 같은 모양으로 렌더링되고, `external` 여부만 표시가 다릅니다.
 */
export type WritingEntry = {
  id: string;
  title: string;
  date: Date;
  tags: string[];
  href: string;
  /** true 면 외부 링크 — 새 탭으로 열고 출처를 표시합니다. */
  external: boolean;
  /** 외부 글의 출처 이름. 예: velog */
  source?: string;
  /** 내부 글의 한 줄 요약, 또는 외부 글의 메모. */
  description?: string;
};

/** dev 서버에서는 draft 도 보여주고, 빌드에서는 제외합니다. */
const showDrafts = import.meta.env.DEV;

export async function getWriting(): Promise<WritingEntry[]> {
  const [posts, links] = await Promise.all([
    getCollection('posts', ({ data }) => showDrafts || !data.draft),
    getCollection('links'),
  ]);

  const internal: WritingEntry[] = posts.map((post) => ({
    id: post.id,
    title: post.data.title,
    date: post.data.date,
    tags: post.data.tags,
    href: `/posts/${post.id}/`,
    external: false,
    description: post.data.description,
  }));

  const external: WritingEntry[] = links.map((link) => ({
    id: link.id,
    title: link.data.title,
    date: link.data.date,
    tags: link.data.tags,
    href: link.data.url,
    external: true,
    source: link.data.source,
    description: link.data.note,
  }));

  return [...internal, ...external].sort(
    (a, b) => b.date.getTime() - a.date.getTime(),
  );
}

/** 태그 → 글 목록. 내부·외부 글이 함께 잡힙니다. */
export async function getWritingByTag(): Promise<Map<string, WritingEntry[]>> {
  const all = await getWriting();
  const byTag = new Map<string, WritingEntry[]>();

  for (const entry of all) {
    for (const tag of entry.tags) {
      const bucket = byTag.get(tag);
      if (bucket) bucket.push(entry);
      else byTag.set(tag, [entry]);
    }
  }

  return new Map(
    [...byTag.entries()].sort((a, b) => {
      const byCount = b[1].length - a[1].length;
      return byCount !== 0 ? byCount : a[0].localeCompare(b[0]);
    }),
  );
}

/**
 * 태그를 URL 로 쓸 수 있게 바꿉니다. "Contrastive Learning" → "contrastive-learning".
 * 한글 태그는 그대로 두고 공백만 정리합니다.
 */
export function tagSlug(tag: string): string {
  return tag
    .trim()
    .toLowerCase()
    .replace(/[\s/]+/g, '-')
    .replace(/[^\p{Letter}\p{Number}-]/gu, '')
    .replace(/-{2,}/g, '-')
    .replace(/^-|-$/g, '');
}

/** 태그 링크 경로. */
export function tagHref(tag: string): string {
  return `/tags/${encodeURIComponent(tagSlug(tag))}/`;
}

/** 목록에 쓰는 날짜 표기. 2026.08.24 */
export function formatDate(date: Date): string {
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');
  return `${yyyy}.${mm}.${dd}`;
}
