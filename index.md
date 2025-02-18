---
layout: default
title: Home
---

# 🚀 내 블로그에 오신 것을 환영합니다!
이곳은 GitHub Pages + Jekyll을 이용한 블로그입니다.

## 📝 최근 글
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: "%Y-%m-%d" }})
    </li>
  {% endfor %}
</ul>
