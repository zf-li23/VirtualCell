import { useState, useEffect, useRef, useReducer } from 'react'

interface TocItem {
  id: string
  text: string
  level: number
}

interface TocState {
  items: TocItem[]
}

type TocAction =
  | { type: 'setItems'; items: TocItem[] }

function tocReducer(_state: TocState, action: TocAction): TocState {
  switch (action.type) {
    case 'setItems':
      return { items: action.items }
  }
}

interface TableOfContentsProps {
  /** CSS selector for the container that holds the rendered markdown headings */
  containerSelector?: string
}

/** 将标题文本转为 kebab-case 的 ID（与 Markdown.tsx 中的保持一致） */
function headingToId(text: string): string {
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\w\u4e00-\u9fff\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

/** 扫描 DOM 构建 TOC 条目 */
function scanTocItems(containerSelector: string): TocItem[] {
  const container = document.querySelector(containerSelector)
  if (!container) return []

  const headings = container.querySelectorAll<HTMLElement>('h1, h2, h3, h4')
  const items: TocItem[] = []

  headings.forEach((h) => {
    const level = parseInt(h.tagName[1], 10)
    const text = h.textContent ?? ''
    const id = h.id || headingToId(text)

    if (!h.id) {
      h.id = id
    }

    if (id === '学习路线图' || text.includes('学习路线图')) return

    items.push({ id, text, level })
  })

  return items
}

export function TableOfContents({
  containerSelector = '.markdown-body',
}: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>('')
  const [{ items }, dispatch] = useReducer(tocReducer, { items: [] })
  const observerRef = useRef<IntersectionObserver | null>(null)

  // 扫描容器中的标题元素，构建 TOC 条目
  useEffect(() => {
    dispatch({ type: 'setItems', items: scanTocItems(containerSelector) })
  }, [containerSelector])

  // IntersectionObserver 追踪当前可见标题
  useEffect(() => {
    if (items.length === 0) return

    // 清理旧 observer
    if (observerRef.current) {
      observerRef.current.disconnect()
    }

    const observer = new IntersectionObserver(
      (entries) => {
        // 找到第一个进入视口的标题
        const visible = entries.filter((e) => e.isIntersecting)
        if (visible.length > 0) {
          // 取最靠上的可见标题
          visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
          setActiveId(visible[0].target.id)
        }
      },
      {
        rootMargin: '-60px 0px -80% 0px',
        threshold: 0,
      },
    )

    items.forEach((item) => {
      const el = document.getElementById(item.id)
      if (el) observer.observe(el)
    })

    observerRef.current = observer

    return () => {
      observer.disconnect()
    }
  }, [items])

  if (items.length < 2) return null

  return (
    <nav className="toc">
      <div className="toc-header">目录</div>
      <ul className="toc-list">
        {items.map((item) => (
          <li
            key={item.id}
            className={`toc-item toc-level-${item.level} ${activeId === item.id ? 'toc-active' : ''}`}
          >
            <a
              href={`#${item.id}`}
              onClick={(e) => {
                e.preventDefault()
                const el = document.getElementById(item.id)
                if (el) {
                  el.scrollIntoView({ behavior: 'smooth', block: 'start' })
                  const url = new URL(window.location.href)
                  url.hash = item.id
                  window.history.replaceState(null, '', url.toString())
                }
              }}
            >
              {item.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
