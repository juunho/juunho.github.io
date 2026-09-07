import type { APIRoute } from 'astro';

/**
 * 어드민 (Sveltia CMS). https://juunho.github.io/admin/
 *
 * 어디서든 — 다른 PC, 폰 — 글을 쓸 수 있게 배포합니다.
 *
 * ## 접근에 대해
 *
 * GitHub Pages 는 정적 호스팅이라 배포된 파일 앞에 인증을 세울 수 없습니다.
 * 그래서 이 URL 자체는 누구나 열 수 있습니다. 다만 **열어도 아무것도 없습니다** —
 * 이 페이지는 빈 껍데기이고, 이 저장소에 쓰기 권한이 있는 GitHub 자격증명이
 * 없으면 로그인 화면에서 더 나아가지 못합니다. 초안을 읽을 수도, 무언가를
 * 쓸 수도 없습니다. 아래 config 가 드러내는 것은 폴더 이름뿐인데, 저장소가
 * 어차피 공개라 새로 새는 정보도 없습니다.
 *
 * URL 존재 자체를 감추려면 GitHub Pages 를 벗어나야 합니다
 * (예: Cloudflare Pages + Cloudflare Access).
 *
 * ## 로그인
 *
 * 로그인 화면의 **Sign In with Token** 을 누르면 필요한 권한이 미리 선택된
 * GitHub 토큰 발급 링크가 뜹니다. fine-grained 토큰을 이 저장소 하나로,
 * Contents: Read and write 만 주고 만드세요. 토큰은 그 브라우저에만 저장되고
 * 저장소에는 절대 들어가지 않습니다.
 *
 * "Sign in with GitHub" 버튼(OAuth)으로 바꾸려면 인증용 서버가 하나 필요합니다.
 * sveltia-cms-auth 를 Cloudflare Workers 무료 플랜에 올리고, 아래 backend 에
 * base_url 한 줄을 추가하면 됩니다.
 *
 * ## 로컬에서 쓰기
 *
 * npm run dev → http://localhost:4321/admin/index.html 을 Chrome/Edge/Brave 로
 * 열면 토큰 없이 로컬 파일을 직접 고치는 모드로 동작합니다 (File System
 * Access API). 배포본과 같은 설정을 그대로 씁니다.
 */

export function getStaticPaths() {
  // index.html 을 명시적으로 씁니다. slug 없이 /admin 을 함께 등록하면
  // 두 경로가 같은 출력 파일로 겹쳐서 빌드가 막힙니다.
  return [{ params: { slug: 'index.html' } }, { params: { slug: 'config.yml' } }];
}

const SHELL = `<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="robots" content="noindex, nofollow" />
    <title>juunho — 어드민</title>
    <link href="/admin/config.yml" type="text/yaml" rel="cms-config-url" />
  </head>
  <body>
    <script src="https://unpkg.com/@sveltia/cms/dist/sveltia-cms.js"><\/script>
  </body>
</html>
`;

/**
 * Sveltia 설정. 위젯 이름과 필드는 src/content.config.ts 의 스키마와
 * 맞춰야 합니다 — 한쪽만 고치면 저장한 파일이 스키마 검증에서 걸립니다.
 */
const CONFIG = `# 이 파일은 src/pages/admin/[...slug].ts 가 dev 서버에서만 생성합니다.
# 고치려면 그 파일을 여세요.

backend:
  name: github
  repo: juunho/juunho.github.io
  branch: main
  commit_messages:
    create: 'admin: {{collection}} 추가 — {{slug}}'
    update: 'admin: {{collection}} 수정 — {{slug}}'
    delete: 'admin: {{collection}} 삭제 — {{slug}}'
    uploadMedia: 'admin: 파일 업로드 — {{path}}'
    deleteMedia: 'admin: 파일 삭제 — {{path}}'
  # OAuth("Sign in with GitHub" 버튼)로 바꾸려면 인증 서버 주소를 여기에.
  # 없으면 토큰 로그인만 쓰며, 그것만으로도 충분히 동작합니다.
  # base_url: https://<your-worker>.workers.dev

media_folder: public/uploads
public_folder: /uploads

# 커밋 메시지에서 CMS 로 쓴 것임을 알 수 있게
slug:
  encoding: unicode
  clean_accents: true

collections:
  # ── 여기에 직접 쓰는 글 ──────────────────────────────────────
  - name: posts
    label: 글
    label_singular: 글
    folder: src/content/posts
    create: true
    extension: md
    format: yaml-frontmatter
    slug: '{{slug}}'
    summary: '{{title}}'
    sortable_fields: [date, title]
    view_groups:
      - label: 상태
        field: draft
    fields:
      - { name: title, label: 제목, widget: string }
      - {
          name: date,
          label: 날짜,
          widget: datetime,
          date_format: 'YYYY-MM-DD',
          time_format: false,
          picker_utc: true,
        }
      - {
          name: description,
          label: 한 줄 요약,
          widget: text,
          required: false,
          hint: '목록과 검색 결과, 그리고 링크 미리보기에 쓰입니다.',
        }
      - {
          name: tags,
          label: 태그,
          widget: list,
          default: [],
          hint: '첫 번째 태그가 목록 오른쪽에 표시됩니다.',
        }
      - {
          name: draft,
          label: 초안,
          widget: boolean,
          default: false,
          hint: '켜두면 배포에 포함되지 않습니다. 로컬에서는 보입니다.',
        }
      - {
          name: body,
          label: 본문,
          widget: markdown,
          hint: '수식은 $x$ 또는 $$ ... $$ 로 씁니다.',
        }

  # ── 다른 데 쓴 글 (북마크) ───────────────────────────────────
  - name: links
    label: 다른 데 쓴 글
    files:
      - name: links
        label: 북마크 목록
        file: src/content/links.yaml
        description: 'velog, Medium 등 외부에 쓴 글. 글 목록에 같이 섞여 나옵니다.'
        fields:
          - name: links
            label: 링크
            label_singular: 링크
            widget: list
            summary: '{{fields.title}} — {{fields.source}}'
            fields:
              - { name: title, label: 제목, widget: string }
              - {
                  name: date,
                  label: 날짜,
                  widget: datetime,
                  date_format: 'YYYY-MM-DD',
                  time_format: false,
                  picker_utc: true,
                }
              - { name: url, label: 원본 주소, widget: string }
              - {
                  name: source,
                  label: 출처,
                  widget: string,
                  hint: '목록에 "↗ velog" 처럼 표시됩니다.',
                }
              - { name: tags, label: 태그, widget: list, default: [] }
              - {
                  name: note,
                  label: 메모,
                  widget: text,
                  required: false,
                  hint: '왜 걸어뒀는지 한 줄. 비워도 됩니다.',
                }

  # ── 해온 일 ─────────────────────────────────────────────────
  - name: work
    label: 해온 일
    label_singular: 이력
    folder: src/content/work
    create: true
    extension: md
    format: yaml-frontmatter
    slug: '{{slug}}'
    summary: '{{title}} — {{period}}'
    sortable_fields: [order, title]
    fields:
      - { name: title, label: 제목, widget: string }
      - {
          name: kind,
          label: 분류,
          widget: string,
          hint: '제목 옆에 작게 붙습니다. 예: Retrieval, Training, Serving, Open Source',
        }
      - {
          name: period,
          label: 기간,
          widget: string,
          hint: '왼쪽 컬럼에 그대로 나옵니다. 예: 2025 — 현재',
        }
      - {
          name: order,
          label: 정렬 순서,
          widget: number,
          value_type: int,
          hint: '큰 값이 위로 옵니다.',
        }
      - { name: summary, label: 설명, widget: text }
      - { name: stack, label: 스택, widget: list, default: [] }
      - {
          name: link,
          label: 링크,
          widget: object,
          required: false,
          collapsed: true,
          fields:
            [
              { name: href, label: 주소, widget: string },
              { name: label, label: 표시할 문구, widget: string },
            ],
        }
`;

export const GET: APIRoute = ({ params }) => {
  if (params.slug === 'config.yml') {
    return new Response(CONFIG, {
      headers: { 'content-type': 'text/yaml; charset=utf-8' },
    });
  }

  return new Response(SHELL, {
    headers: {
      'content-type': 'text/html; charset=utf-8',
      'x-robots-tag': 'noindex, nofollow',
    },
  });
};
