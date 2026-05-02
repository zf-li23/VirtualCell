import type { NoteMeta } from '../config/notes'

interface SidebarProps {
  notesByCategory: Map<string, NoteMeta[]>
  activeId: string
  search: string
  onSearch: (val: string) => void
  onSelect: (id: string) => void
}

export function Sidebar({
  notesByCategory,
  activeId,
  search,
  onSearch,
  onSelect,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>🧬 VirtualCell</h1>
        <p className="subtitle">笔记浏览器</p>
      </div>
      <div className="search-wrap">
        <input
          type="text"
          className="search-input"
          placeholder="搜索笔记..."
          value={search}
          onChange={(e) => onSearch(e.target.value)}
        />
      </div>
      <nav className="nav">
        {Array.from(notesByCategory.entries()).map(([cat, list]) => (
          <div key={cat} className="nav-group">
            <div className="nav-category">{cat}</div>
            <ul>
              {list.map((note) => (
                <li key={note.id}>
                  <button
                    className={note.id === activeId ? 'active' : ''}
                    onClick={() => onSelect(note.id)}
                  >
                    {note.title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
        {notesByCategory.size === 0 && (
          <div className="empty">未找到匹配的笔记</div>
        )}
      </nav>
    </aside>
  )
}
