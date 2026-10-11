/** 사이트 전역 설정. 여기만 고치면 사이트 전체에 반영됩니다. */

export const SITE = {
  url: 'https://juunho.github.io',
  title: 'juunho',
  /** 홈 마스트헤드에 크게 들어가는 두 줄. 둘째 줄에 강조색이 들어갑니다. */
  headline: ['I build search', 'that works.'],
  /** 마스트헤드 아래 소개 문단. */
  bio: 'I work on the retrieval side of RAG systems: training embedding models, blending lexical and dense search, and keeping it fast under real traffic. Sometimes I write things down.',
  /** <meta name="description"> 및 RSS 설명. */
  description:
    'Notes on retrieval, embeddings, and inference — the work and writing of juunho.',
  /** 사이트 UI 언어. 글은 frontmatter 의 lang 으로 따로 지정할 수 있습니다. */
  lang: 'en',
  author: 'juunho',
  email: 'juunho.bae@gmail.com',
} as const;

export const NAV = [
  { href: '/', label: 'Home' },
  { href: '/writing/', label: 'Writing' },
  { href: '/tags/', label: 'Tags' },
  { href: '/about/', label: 'About' },
  { href: '/search/', label: 'Search' },
] as const;

export const SOCIAL = [
  { href: 'https://github.com/juunho', label: 'GitHub' },
  { href: `mailto:${SITE.email}`, label: SITE.email },
] as const;

/**
 * giscus 댓글. GitHub Discussions 기반.
 *
 * 켜는 법:
 *  1. 레포 Settings → Features → Discussions 체크
 *  2. https://github.com/apps/giscus 설치
 *  3. https://giscus.app/ko 에서 레포를 넣고 나오는 값을 아래에 채우기
 *  4. enabled 를 true 로
 *
 * repoId / categoryId 가 비어 있으면 댓글 영역은 그냥 렌더링되지 않습니다.
 */
export const GISCUS = {
  enabled: false,
  repo: 'juunho/juunho.github.io',
  repoId: '',
  category: 'Announcements',
  categoryId: '',
} as const;

/** 홈에 보여줄 글 개수. */
export const HOME_WRITING_LIMIT = 5;
