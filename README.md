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

## 어드민

<https://juunho.github.io/admin/> — 폼으로 글을 쓰고 이력·북마크를 고칩니다.
[Sveltia CMS](https://github.com/sveltia/sveltia-cms) 를 씁니다.
저장하면 이 저장소에 커밋되고, Actions 가 빌드해서 배포합니다 (1~2분).

### 처음 한 번: 토큰 만들기

로그인 화면에서 **Sign In with Token** 을 누르면 GitHub 토큰 발급 페이지
링크가 뜹니다 (필요한 권한이 미리 선택되어 있습니다).

fine-grained 토큰으로 만들되 범위를 좁히세요:

- **Repository access** — `juunho/juunho.github.io` 하나만
- **Permissions → Contents** — Read and write
- 만료일을 짧게 두고 만료되면 새로 발급

토큰은 그 브라우저에만 저장되고 저장소에는 들어가지 않습니다. 기기마다
한 번씩 넣으면 됩니다. 공용 PC 에서는 쓰지 마세요.

### 접근에 대해 — 알고 계셔야 할 것

**`/admin/` URL 은 누구나 열 수 있습니다.** GitHub Pages 는 정적 호스팅이라
배포된 파일 앞에 인증을 세울 수 없습니다.

다만 **열어도 아무것도 없습니다.** 이 페이지는 빈 껍데기라, 이 저장소에
쓰기 권한이 있는 GitHub 자격증명이 없으면 로그인 화면에서 막힙니다.
초안을 읽을 수도, 무언가를 쓸 수도 없습니다. `config.yml` 이 드러내는 것은
폴더 이름뿐인데 저장소가 어차피 공개라 새로 새는 정보도 없습니다.

`robots.txt` 와 사이트맵에서 빼서 검색 결과에는 안 나오지만, 그건 가리는
것이지 막는 게 아닙니다. 실제로 막는 것은 GitHub 로그인입니다.

URL 존재 자체를 감추려면 GitHub Pages 를 벗어나야 합니다
(예: Cloudflare Pages + Cloudflare Access — 무료 플랜으로 경로 앞에 진짜
인증을 세울 수 있습니다).

### 로그인 버튼으로 바꾸기 (선택)

토큰을 붙여넣는 대신 "Sign in with GitHub" 버튼을 쓰려면 인증 서버가
하나 필요합니다. [sveltia-cms-auth](https://github.com/sveltia/sveltia-cms-auth)
를 Cloudflare Workers 무료 플랜에 올리고,
`public/admin/config.yml` 의 `backend` 에 `base_url` 한 줄을 추가하면
됩니다 (주석으로 자리를 남겨뒀습니다).

### 폼 자체를 고치기

어드민의 폼은 **`public/admin/config.yml`** 이 정의합니다. 이 파일을 고치면
폼이 바뀝니다. 파일 맨 위 주석에 무엇을 고쳐도 되는지 정리해뒀습니다.

**어디서 고치나** — github.com 에서 이 파일을 열고 연필 아이콘을 누르면
됩니다. 폰에서도 되고 문법 강조도 나옵니다. 또는 로컬에서 고치고 push.

**어드민 화면 안에서는 못 고칩니다.** Sveltia 에 설정 편집 UI 가 아직
없습니다 (로드맵에는 [올라와 있습니다](https://github.com/sveltia/sveltia-cms/discussions/452)).

바로 고쳐도 되는 것:

| | |
|---|---|
| `label`, `label_singular` | 사이드바와 버튼에 보이는 이름 (`글` → `포스트`) |
| `description`, `hint` | 폼에 뜨는 설명 문구 |
| `fields` 순서 | 입력칸 위아래 |
| `required`, `default`, `collapsed` | 입력 편의 |
| `sortable_fields`, `summary`, `view_groups` | 목록 화면 |

**필드를 새로 추가하는 것도 안전합니다.** 사이트 빌드는 모르는 항목을
조용히 무시하므로 깨지지 않습니다. 다만 그 값이 화면에 나오려면 두 곳을
더 고쳐야 합니다 — `src/content.config.ts` 의 스키마, 그리고 실제로
출력할 컴포넌트.

**필드를 지울 때는** `src/content.config.ts` 에서 필수로 잡힌 항목인지
먼저 확인하세요. 필수 항목을 폼에서 없애면 그 뒤로 저장한 글이 빌드에서
걸립니다.

YAML 문법이 깨지면 어드민이 열리지 않지만, CI 가 배포 전에 잡아서 빌드를
실패시킵니다. 이미 커밋했다면 github.com 에서 되돌리면 살아납니다.

### 로컬에서 쓰기

`npm run dev` 후 Chrome / Edge / Brave 로
<http://localhost:4321/admin/index.html> 을 열면 **토큰 없이** 로컬 파일을
직접 고치는 모드로 동작합니다 (File System Access API). 폴더 선택에서
이 저장소 루트를 고르면 됩니다. 저장하면 로컬 파일이 바뀌므로
`git push` 는 직접 해야 합니다. Firefox 와 Safari 는 이 API 가 없어
로컬 모드가 동작하지 않습니다.

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
  pages/               홈, /writing, /tags, /about, /search, rss.xml
  styles/global.css    색 토큰과 타이포
```

어드민은 `public/admin/` 에 있습니다 — `index.html` 이 화면,
`config.yml` 이 폼 정의입니다. 위의 **폼 자체를 고치기** 를 참고하세요.
