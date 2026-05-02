import { useState, useMemo, useEffect } from 'react'
import { noteMetas, categories, type NoteMeta } from '../config/notes'
import { noteLoaders } from '../lib/loaders'

export interface LoadedNote extends NoteMeta {
  content: string
}

export function useNotes() {
  const [activeId, setActiveId] = useState('overview')
  const [search, setSearch] = useState('')
  const [cache, setCache] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
    if (cache[activeId]) {
      setLoading(false)
      return
    }
    const loader = noteLoaders[activeId]
    if (!loader) return

    let cancelled = false
    setLoading(true)
    setError(null)
    loader()
      .then((content) => {
        if (cancelled) return
        setCache((prev) => ({ ...prev, [activeId]: content }))
        setLoading(false)
      })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [activeId])

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
