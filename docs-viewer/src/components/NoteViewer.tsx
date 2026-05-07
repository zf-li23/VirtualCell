import type { LoadedNote } from '../hooks/useNotes'
import { Markdown } from './Markdown'
import { LearningRoadmap } from './Roadmap'
import { TableOfContents } from './TableOfContents'

interface NoteViewerProps {
  note: LoadedNote | null
  loading: boolean
  error: string | null
  onNavigate?: (id: string) => void
}

export function NoteViewer({ note, loading, error, onNavigate }: NoteViewerProps) {
  return (
    <main className="main">
      <header className="main-header">
        <h2>{note?.title ?? '未选择'}</h2>
        {note && <span className="badge">{note.category}</span>}
      </header>
      <div className="content-with-toc">
        <article className="markdown-body">
          {loading && <div className="status">加载中...</div>}
          {error && <div className="status error">{error}</div>}
          {!loading && !error && note && (
            <>
              {note.id === 'overview' && onNavigate && <LearningRoadmap onNavigate={onNavigate} />}
              <Markdown content={note.content} onNavigate={onNavigate} />
            </>
          )}
        </article>
        {!loading && !error && note && note.id !== 'overview' && (
          <TableOfContents key={note.id} />
        )}
      </div>
    </main>
  )
}
