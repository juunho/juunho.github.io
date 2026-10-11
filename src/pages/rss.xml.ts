import rss from '@astrojs/rss';
import type { APIContext } from 'astro';

import { SITE } from '../consts';
import { getWriting } from '../lib/writing';
import { getSiteCopy } from '../lib/site';

export async function GET(context: APIContext) {
  const [entries, copy] = await Promise.all([getWriting(), getSiteCopy()]);
  const site = context.site ?? new URL(SITE.url);

  return rss({
    title: SITE.title,
    description: copy.description,
    site,
    customData: `<language>${SITE.lang}</language>`,
    items: entries.map((entry) => ({
      title: entry.title,
      pubDate: entry.date,
      description: entry.description,
      categories: entry.tags,
      // 외부 글은 원본으로 바로 보냅니다.
      link: entry.external ? entry.href : new URL(entry.href, site).toString(),
    })),
  });
}
