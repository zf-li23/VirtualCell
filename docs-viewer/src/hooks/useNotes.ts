import { useState, useMemo, useEffect, useReducer, useRef } from 'react'
import { noteMetas, categories, type NoteMeta } from '../config/notes'
import { noteLoaders } from '../lib/loaders'

export interface LoadedNote extends NoteMeta {
  content: string
}

interface FetchState {
  cache: Record<string, string>
  fetching: boolean
  error: string | null
}

type FetchAction =
  | { type: 'fetchStart' }
  | { type: 'fetchSuccess'; id: string; content: string }
  | { type: 'fetchError'; error: string }

function fetchReducer(state: FetchState, action: FetchAction): FetchState {
  switch (action.type) {
    case 'fetchStart':
      return { ...state, fetching: true, error: null }
    case 'fetchSuccess':
      return {
        ...state,
        fetching: false,
        cache: { ...state.cache, [action.id]: action.content },
      }
    case 'fetchError':
      return { ...state, fetching: false, error: action.error }
  }
}

/** 从 URL search param 读取初始笔记 ID */
function getInitialNoteId(): string {
  if (typeof window === 'undefined') return 'overview'
  const params = new URLSearchParams(window.location.search)
  const noteFromUrl = params.get('note')
  if (noteFromUrl && noteLoaders[noteFromUrl]) return noteFromUrl
  return 'overview'
}

export function useNotes() {
  const [activeId, setActiveId] = useState(getInitialNoteId)
  const [search, setSearch] = useState('')
  const [{ cache, fetching, error }, dispatch] = useReducer(fetchReducer, {
    cache: {},
    fetching: false,
    error: null,
  })
  const isInitialMount = useRef(true)

  // loading is derived: true only while actively fetching a note not yet cached
  const loading = fetching && !cache[activeId]

  const filtered = useMemo(() => {
    if (!search.trim()) return noteMetas
    const s = search.toLowerCase()
    return noteMetas.filter(
      (n) =>
        n.title.toLowerCase().includes(s) ||
        n.category.toLowerCase().includes(s)
    )
  }, [search])

  const notesByCategory = useMemo(() => {
    const map = new Map<string, NoteMeta[]>()
    for (const cat of categories) {
      const list = filtered.filter((n) => n.category === cat)
      if (list.length) map.set(cat, list)
    }
    return map
  }, [filtered])

  const activeNote: LoadedNote | null = useMemo(() => {
    const meta = noteMetas.find((n) => n.id === activeId)
    if (!meta) return null
    return { ...meta, content: cache[activeId] ?? '' }
  }, [activeId, cache])

  useEffect(() => {
    // Already cached — no fetch needed
    // Intentionally not including `cache` in deps: adding it would cause the
    // effect to re-run on every cache update, creating an infinite loop.
    if (cache[activeId]) return

    const loader = noteLoaders[activeId]
    if (!loader) return

    let cancelled = false
    dispatch({ type: 'fetchStart' })
    loader()
      .then((content) => {
        if (cancelled) return
        dispatch({ type: 'fetchSuccess', id: activeId, content })
      })
      .catch((err) => {
        if (cancelled) return
        dispatch({
          type: 'fetchError',
          error: err instanceof Error ? err.message : String(err),
        })
      })

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId])

  // === URL 双向同步 ===

  // activeId → URL: 笔记切换时更新 ?note= 参数
  useEffect(() => {
    const url = new URL(window.location.href)
    if (activeId === 'overview') {
      url.searchParams.delete('note')
    } else {
      url.searchParams.set('note', activeId)
    }
    // 非首次加载时清空 hash（旧笔记的锚点不适用新笔记）
    if (!isInitialMount.current) {
      url.hash = ''
    }
    isInitialMount.current = false
    window.history.replaceState(null, '', url.toString())
  }, [activeId])

  // 笔记加载完成后，若 URL 有 hash 则滚动到对应标题
  useEffect(() => {
    if (loading || error || !cache[activeId]) return
    const hash = window.location.hash
    if (!hash || hash.length <= 1) return

    const id = decodeURIComponent(hash.slice(1))
    // 等待 DOM 渲染后滚动
    requestAnimationFrame(() => {
      const el = document.getElementById(id)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, activeId])

  return {
    activeNote,
    notesByCategory,
    search,
    setSearch,
    activeId,
    setActiveId,
    loading,
    error,
  }
}
