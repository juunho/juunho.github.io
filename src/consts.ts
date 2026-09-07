/** 사이트 전역 설정. 여기만 고치면 사이트 전체에 반영됩니다. */

export const SITE = {
  url: 'https://juunho.github.io',
  title: 'juunho',
  /** 홈 마스트헤드에 크게 들어가는 두 줄. */
  headline: ['검색이 잘 되게', '만드는 사람'],
  /** 마스트헤드 아래 소개 문단. */
  bio: 'RAG 파이프라인의 검색 단을 설계하고, 임베딩 모델을 학습시키고, 그걸 실제 트래픽 위에서 버티게 만듭니다. 배운 건 가끔 글로 남깁니다.',
  /** <meta name="description"> 및 RSS 설명. */
  description: '검색·임베딩·추론 최적화에 대한 기록. juunho의 작업과 글.',
  lang: 'ko',
  author: 'juunho',
  email: 'juunho.bae@gmail.com',
} as const;

export const NAV = [
  { href: '/', label: '홈' },
  { href: '/writing/', label: '글' },
  { href: '/tags/', label: '태그' },
  { href: '/about/', label: '소개' },
  { href: '/search/', label: '검색' },
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
