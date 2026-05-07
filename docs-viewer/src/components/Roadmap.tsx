import { useState, useEffect } from 'react'
import { roadmapRoutes, type RoadmapRoute } from '../config/roadmap'

interface RoadmapProps {
  onNavigate: (id: string) => void
}

const STORAGE_KEY = 'roadmap-progress'

function loadProgress(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
  } catch {
    return {}
  }
}

function saveProgress(p: Record<string, boolean>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(p))
}

export function LearningRoadmap({ onNavigate }: RoadmapProps) {
  const [progress, setProgress] = useState<Record<string, boolean>>(loadProgress)

  useEffect(() => {
    saveProgress(progress)
  }, [progress])

  const toggle = (noteId: string) => {
    setProgress((prev) => ({ ...prev, [noteId]: !prev[noteId] }))
  }

  const totalItems = roadmapRoutes.reduce((s, r) => s + r.items.length, 0)
  const doneItems = roadmapRoutes.reduce(
    (s, r) => s + r.items.filter((i) => progress[i.noteId]).length,
    0,
  )

  return (
    <div className="roadmap">
      {/* 进度条 */}
      <div className="roadmap-header">
        <h3>🧭 学习路线图</h3>
        <span className="roadmap-progress-text">
          {doneItems} / {totalItems} 已完成
        </span>
      </div>
      <div className="roadmap-progress-bar-wrap">
        <div
          className="roadmap-progress-bar"
          style={{ width: `${Math.round((doneItems / totalItems) * 100)}%` }}
        />
      </div>

      {/* 路线卡片 */}
      <div className="roadmap-routes">
        {roadmapRoutes.map((route) => (
          <RouteCard
            key={route.title}
            route={route}
            progress={progress}
            onToggle={toggle}
            onNavigate={onNavigate}
          />
        ))}
      </div>
    </div>
  )
}

function RouteCard({
  route,
  progress,
  onToggle,
  onNavigate,
}: {
  route: RoadmapRoute
  progress: Record<string, boolean>
  onToggle: (id: string) => void
  onNavigate: (id: string) => void
}) {
  const done = route.items.filter((i) => progress[i.noteId]).length
  const total = route.items.length

  return (
    <div className="roadmap-route">
      <div className="roadmap-route-header">
        <div>
          <h4 className="roadmap-route-title">{route.title}</h4>
          <p className="roadmap-route-desc">{route.description}</p>
        </div>
        <span className="roadmap-route-count">
          {done}/{total}
        </span>
      </div>
      <ul className="roadmap-item-list">
        {route.items.map((item) => {
          const checked = !!progress[item.noteId]
          return (
            <li key={item.noteId} className="roadmap-item">
              <label className="roadmap-checkbox-label">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onToggle(item.noteId)}
                  className="roadmap-checkbox"
                />
                <span
                  className={`roadmap-checkmark ${checked ? 'checked' : ''}`}
                />
              </label>
              <button
                className={`roadmap-link ${checked ? 'done' : ''} ${item.optional ? 'optional' : ''}`}
                onClick={() => onNavigate(item.noteId)}
              >
                {item.label}
                {item.optional && <span className="roadmap-badge-opt">可选</span>}
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
