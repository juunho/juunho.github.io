import type { APIRoute } from 'astro';

/**
 * 로컬 전용 어드민 (Sveltia CMS).
 *
 * **이 경로는 배포되지 않습니다.** getStaticPaths 가 프로덕션 빌드에서 빈
 * 배열을 돌려주기 때문에 dist/ 에 admin 관련 파일이 아예 생기지 않습니다.
 * GitHub Pages 는 정적 호스팅이라 배포된 파일에 비밀번호를 걸 수 없으므로,
 * "나만 접근" 을 만족시키는 방법은 애초에 내보내지 않는 것뿐입니다.
 *
 * 쓰는 법:
 *   npm run dev  →  http://localhost:4321/admin/index.html
 *   (Chrome / Edge / Brave — Firefox 와 Safari 는 안 됩니다)
 *
 * 처음 열면 브라우저가 폴더 선택을 요청합니다. 이 저장소 루트를 고르면
 * File System Access API 로 파일을 직접 읽고 씁니다. 토큰도, 로그인도,
 * 프록시 서버도 필요 없습니다. 저장하면 로컬 파일이 바뀌고, 평소처럼
 * git commit && git push 하면 배포됩니다.
 *
 * Firefox 와 Safari 는 File System Access API 가 없어서 동작하지 않습니다.
 */

export function getStaticPaths() {
  // 프로덕션 빌드에서는 경로가 하나도 생성되지 않습니다.
  if (!import.meta.env.DEV) return [];

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
    <title>juunho — 어드민 (로컬 전용)</title>
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

# 로컬 모드에서는 이 backend 설정을 쓰지 않지만, 항목 자체는 있어야 합니다.
backend:
  name: github
  repo: juunho/juunho.github.io
  branch: main

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
