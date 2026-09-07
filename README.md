# juunho.github.io

포트폴리오 + 기술 블로그. [Astro](https://astro.build) 로 빌드하고 GitHub Actions 가
GitHub Pages 로 배포합니다.

## 로컬에서 돌리기

```bash
npm install
npm run dev        # http://localhost:4321
```

검색(Pagefind)은 빌드 후에만 동작합니다.

```bash
npm run build      # astro build + 검색 인덱스 생성
npm run preview    # 빌드 결과를 그대로 확인
```

## 어드민 (로컬 전용)

폼으로 글을 쓰고 이력·북마크를 고칠 수 있는 화면이 있습니다.

```bash
npm run dev
```

Chrome / Edge / Brave 로 **<http://localhost:4321/admin/index.html>** 을 엽니다.
처음 열면 브라우저가 폴더를 물어보는데, 이 저장소 루트를 고르면 됩니다.
저장하면 `src/content/` 아래 파일이 실제로 바뀌고, 평소처럼
`git commit && git push` 하면 배포됩니다.

로그인도 토큰도 필요 없습니다. [Sveltia CMS](https://github.com/sveltia/sveltia-cms) 가
브라우저의 File System Access API 로 로컬 파일을 직접 다루기 때문입니다.
그래서 **Firefox 와 Safari 에서는 동작하지 않습니다** — 이 API가 없습니다.

> **어드민은 배포되지 않습니다.**
> GitHub Pages 는 정적 호스팅이라 배포된 파일에 비밀번호를 걸 수 없습니다.
> `/admin/` 을 올리면 URL 자체는 누구나 열 수 있게 되므로, 아예 내보내지
> 않는 쪽을 택했습니다. `src/pages/admin/[...slug].ts` 가 dev 에서만 경로를
> 만들고, CI 에도 혹시 섞여 들어가면 빌드를 실패시키는 스텝이 있습니다.
>
> 폰이나 다른 컴퓨터에서도 쓰고 싶어지면 GitHub 토큰 방식으로 바꿀 수
> 있습니다. 그때는 `/admin/` 이 공개 URL 이 되고, 토큰이 없는 사람에게는
> 로그인 화면만 보입니다.

아래는 파일을 직접 고칠 때의 형식입니다. 어드민을 쓰면 폼이 대신 채워줍니다.

## 글 쓰기

### 여기에 직접 쓰는 글

`src/content/posts/` 에 `.md` 파일을 만듭니다. 파일 이름이 그대로 URL 이 됩니다
(`my-post.md` → `/posts/my-post/`).

```markdown
---
title: 글 제목
date: 2026-08-24
description: 목록과 검색 결과에 나오는 한 줄 요약. (선택)
tags: [Embedding, RAG]
draft: false
---

본문. 수식은 `$x$` 처럼 인라인으로, 또는

$$
L = \sum_{i=1}^{N} x_i
$$

처럼 블록으로 씁니다. KaTeX 로 **빌드할 때** 렌더링되기 때문에
브라우저에는 수식 자바스크립트가 실리지 않습니다.
```

`draft: true` 로 두면 `npm run dev` 에서는 보이지만 배포에는 포함되지 않습니다.

### 다른 곳에 쓴 글 (북마크)

velog, Medium, 회사 기술블로그 등에 쓴 글은 `src/content/links.yaml` 에 추가합니다.

```yaml
links:
  - title: "RAG 검색 품질을 어떻게 측정할 것인가"
    date: 2026-07-11
    url: https://velog.io/@juunho/rag-retrieval-eval
    source: velog
    tags: [RAG, Evaluation]
    note: "리랭커 부분만 보셔도 됩니다"   # 선택
```

두 종류는 **하나의 목록으로 합쳐져** 날짜순으로 정렬됩니다. 줄 모양과 서체는
동일하고, 외부 글에만 오른쪽에 `↗ 출처` 가 붙습니다. 태그 아카이브와 RSS 에도
함께 들어갑니다.

## 이력 고치기

`src/content/work/` 의 `.md` 파일 하나가 홈의 "해온 일" 한 줄입니다.
본문은 쓰지 않고 프론트매터만 씁니다.

```markdown
---
title: FAQ 검색 파이프라인
kind: Retrieval          # 제목 옆에 작게 붙는 분류
period: 2025 — 현재       # 왼쪽 기간 컬럼
order: 40                # 큰 값이 위로
summary: >-
  두 줄 정도의 설명.
stack: [Python, FAISS, Triton]
link:                    # 선택
  href: /posts/some-post/
  label: 관련 글
---
```

## 사이트 설정

`src/consts.ts` 한 곳에 모여 있습니다 — 제목, 마스트헤드 문구, 소개글,
네비게이션, 연락처, 홈에 보여줄 글 개수.

### 댓글 켜기

댓글은 [giscus](https://giscus.app/ko) (GitHub Discussions) 를 씁니다.
기본값은 꺼져 있습니다.

1. 레포 Settings → Features → **Discussions** 체크
2. [giscus 앱](https://github.com/apps/giscus) 설치
3. <https://giscus.app/ko> 에서 이 레포를 넣고 나오는 `repoId`, `categoryId` 복사
4. `src/consts.ts` 의 `GISCUS` 에 채우고 `enabled: true`

## 배포

`main` 에 푸시하면 `.github/workflows/deploy.yml` 이 빌드해서 Pages 로 올립니다.

처음 한 번은 레포 **Settings → Pages → Source** 를 **GitHub Actions** 로
바꿔야 합니다.

## 구조

```
src/
  consts.ts            사이트 전역 설정
  content.config.ts    posts / work / links 스키마
  content/
    posts/             여기에 쓴 글 (.md)
    work/              이력 (.md, 프론트매터만)
    links.yaml         다른 데 쓴 글
  lib/
    writing.ts         내부·외부 글을 한 스트림으로 합치는 로직
    katex.ts           빌드 타임 수식 렌더링
  layouts/             Base, Post
  components/          Header, Footer, WritingList, WorkList, …
  pages/
    admin/[...slug].ts 로컬 전용 어드민 (배포 안 됨)
    …                  홈, /writing, /tags, /about, /search, rss.xml
  styles/global.css    색 토큰과 타이포
```

어드민 폼을 고치려면 `src/pages/admin/[...slug].ts` 안의 `CONFIG` 를 고칩니다.
필드는 `src/content.config.ts` 의 스키마와 맞아야 합니다 — 한쪽만 바꾸면
저장한 파일이 스키마 검증에서 걸립니다.
