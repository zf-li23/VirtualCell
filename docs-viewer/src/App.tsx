import { useNotes } from './hooks/useNotes'
import { Sidebar } from './components/Sidebar'
import { NoteViewer } from './components/NoteViewer'
import './App.css'

export default function App() {
  const {
    activeNote,
    notesByCategory,
    search,
    setSearch,
    activeId,
    setActiveId,
    loading,
    error,
  } = useNotes()

  return (
    <div className="app">
      <Sidebar
        notesByCategory={notesByCategory}
        activeId={activeId}
        search={search}
        onSearch={setSearch}
        onSelect={setActiveId}
      />
      <NoteViewer note={activeNote} loading={loading} error={error} onNavigate={setActiveId} />
    </div>
  )
}
