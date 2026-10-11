import { getEntry } from 'astro:content';

/**
 * 사이트 문구 (src/content/site.yaml). 어드민에서 고칠 수 있도록 코드 밖에 둡니다.
 * 파일이 비어 있거나 깨져 있으면 빌드가 스키마 검증에서 멈춥니다 — 조용히 빈
 * 화면을 내보내는 것보다 낫습니다.
 */
export async function getSiteCopy() {
  const entry = await getEntry('site', 'site');
  if (!entry) {
    throw new Error(
      'src/content/site.yaml 을 읽지 못했습니다. 파일이 있는지, YAML 이 올바른지 확인하세요.',
    );
  }
  return entry.data;
}
