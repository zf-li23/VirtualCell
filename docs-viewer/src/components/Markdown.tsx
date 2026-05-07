import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'
import { resolveNoteId, isExternalLink, isAnchorLink } from '../lib/linkResolver'

interface MarkdownProps {
  content: string
  onNavigate?: (id: string) => void
}

/** 将标题文本转为 kebab-case 的 ID（与 Markdown 渲染器生成的一致） */
function headingToId(text: string): string {
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\w\u4e00-\u9fff\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

function getAnchorTarget(href: string): string | null {
  if (!href.startsWith('#')) return null

  try {
    return decodeURIComponent(href.slice(1))
  } catch {
    return href.slice(1)
  }
}

function findAnchorElement(anchor: string): HTMLElement | null {
  const exact = document.getElementById(anchor)
  if (exact) return exact

  const normalizedAnchor = headingToId(anchor)
  if (!normalizedAnchor) return null

  const headings = document.querySelectorAll<HTMLElement>(
    '.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4',
  )

  for (const heading of headings) {
    if (heading.id === normalizedAnchor) return heading
    if (headingToId(heading.textContent ?? '') === normalizedAnchor) return heading
  }

  return null
}

function scrollToAnchor(anchor: string) {
  const el = findAnchorElement(anchor)
  if (!el) return

  el.scrollIntoView({ behavior: 'smooth', block: 'start' })

  const currentHash = window.location.hash.slice(1)
  if (currentHash !== anchor) {
    const url = new URL(window.location.href)
    url.hash = anchor
    window.history.replaceState(null, '', url.toString())
  }
}

export function Markdown({ content, onNavigate }: MarkdownProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{

        // 为标题添加 id，支持锚点跳转
        h1({ children, ...props }) {
          const text = extractText(children)
          const id = headingToId(text)
          return <h1 id={id} {...props}>{children}</h1>
        },
        h2({ children, ...props }) {
          const text = extractText(children)
          const id = headingToId(text)
          return <h2 id={id} {...props}>{children}</h2>
        },
        h3({ children, ...props }) {
          const text = extractText(children)
          const id = headingToId(text)
          return <h3 id={id} {...props}>{children}</h3>
        },
        h4({ children, ...props }) {
          const text = extractText(children)
          const id = headingToId(text)
          return <h4 id={id} {...props}>{children}</h4>
        },

        a({ href, children, ...props }) {
          if (!href) {
            return <span {...props}>{children}</span>
          }

          // 外部链接 → 新标签打开
          if (isExternalLink(href)) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                {...props}
              >
                {children}
              </a>
            )
          }

          // 锚点链接 → 阻止默认跳转，改为平滑滚动
          if (isAnchorLink(href)) {
            return (
              <a
                href={href}
                {...props}
                onClick={(e) => {
                  e.preventDefault()
                  const anchor = getAnchorTarget(href)
                  if (anchor) {
                    scrollToAnchor(anchor)
                  }
                }}
              >
                {children}
              </a>
            )
          }

          // 笔记间相对链接 → 调用 onNavigate
          const noteId = resolveNoteId(href)
          if (noteId && onNavigate) {
            return (
              <a
                href={href}
                {...props}
                onClick={(e) => {
                  e.preventDefault()
                  onNavigate(noteId)
                }}
              >
                {children}
              </a>
            )
          }

          // 无法识别的内部链接，阻止默认导航
          return (
            <a
              href={href}
              {...props}
              onClick={(e) => e.preventDefault()}
              style={{ textDecoration: 'line-through', opacity: 0.6 }}
            >
              {children}
            </a>
          )
        },
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

/** 递归提取 React 子节点的纯文本 */
function extractText(children: React.ReactNode): string {
  if (typeof children === 'string') return children
  if (typeof children === 'number') return String(children)
  if (Array.isArray(children)) return children.map(extractText).join('')
  if (React.isValidElement(children)) {
    const { props } = children as React.ReactElement<{ children?: React.ReactNode }>
    return extractText(props.children)
  }
  return ''
}


