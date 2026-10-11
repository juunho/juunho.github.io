/**
 * 사이트 전역 설정 — 구조에 관한 것만 둡니다.
 * 화면에 보이는 문구(헤드라인, 소개, 링크, About 본문)는 어드민에서 고칠 수
 * 있도록 src/content/site.yaml 과 src/content/pages/about.md 에 있습니다.
 */

export const SITE = {
  url: 'https://juunho.github.io',
  title: 'juunho',
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
