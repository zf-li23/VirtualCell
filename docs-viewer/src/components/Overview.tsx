import { labGroups, type LabGroup } from '../config/researchGroups'
import { techLineage, type TechNode } from '../config/techLineage'
import { roadmapRoutes } from '../config/roadmap'
import { noteMetas, categories } from '../config/notes'
import { LearningRoadmap } from './Roadmap'

interface OverviewProps {
  onNavigate: (id: string) => void
}

/** 统计各分类的笔记数量 */
function categoryCounts() {
  const counts: Record<string, number> = {}
  for (const meta of noteMetas) {
    if (meta.id === 'overview') continue
    counts[meta.category] = (counts[meta.category] || 0) + 1
  }
  return counts
}



export function Overview({ onNavigate }: OverviewProps) {
  const counts = categoryCounts()
  const total = Object.values(counts).reduce((a, b) => a + b, 0)

  return (
    <div className="overview">
      {/* ===== ① 统计总览卡 ===== */}
      <section className="overview-section">
        <h2 className="overview-section-title">📊 统计总览</h2>
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-value">{total}</span>
            <span className="stat-label">总计笔记</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{categories.length}</span>
            <span className="stat-label">分类数</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{labGroups.length}</span>
            <span className="stat-label">课题组</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{techLineage.length}</span>
            <span className="stat-label">技术谱系</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{roadmapRoutes.length}</span>
            <span className="stat-label">学习路线</span>
          </div>
        </div>
        <div className="category-stats">
          {categories.map((cat) => {
            const count = counts[cat] || 0
            if (count === 0) return null
            return (
              <span key={cat} className="category-pill" onClick={() => {
                const first = noteMetas.find((n) => n.category === cat)
                if (first) onNavigate(first.id)
              }}>
                {cat}
                <span className="category-pill-count">{count}</span>
              </span>
            )
          })}
        </div>
      </section>

      {/* ===== ② 技术谱系总览 ===== */}
      <section className="overview-section">
        <h2 className="overview-section-title">🧬 技术谱系总览</h2>
        <p className="overview-section-desc">
          按架构路线组织的模型技术家族。点击模型名可跳转到对应笔记。
        </p>
        <div className="lineage-grid">
          {techLineage.map((tree) => (
            <LineageCard key={tree.name} tree={tree} onNavigate={onNavigate} />
          ))}
        </div>
      </section>

      {/* ===== ③ 课题组关联 ===== */}
      <section className="overview-section">
        <h2 className="overview-section-title">🏫 课题组关联</h2>
        <p className="overview-section-desc">
          同一课题组或机构发布的系列工作，反映了连贯的技术路线与研究积累。
        </p>
        <div className="lab-grid">
          {labGroups.map((group) => (
            <LabCard key={group.name} group={group} onNavigate={onNavigate} />
          ))}
        </div>
      </section>

      {/* ===== ④ 继承关系拓扑 ===== */}
      <section className="overview-section">
        <h2 className="overview-section-title">🔗 继承关系拓扑</h2>
        <p className="overview-section-desc">
          模型之间的直接继承、扩展和演进关系。
        </p>
        <div className="inheritance-list">
          <InheritanceItem from="scGPT" to="scGPT-spatial" desc="空间转录组扩展" />
          <InheritanceItem from="scPRINT" to="scPRINT-2" desc="下一代架构 + lamin.ai 集成" />
          <InheritanceItem from="xTrimoGene" to="scFoundation" desc="非对称 AE → Performer 线性注意力" />
          <InheritanceItem from="Tahoe-100M" to="Tahoe-x1" desc="10B → 30B 参数规模扩展" />
          <InheritanceItem from="scGPT" to="scMulan" desc="GPT 架构 → 多任务生成式" />
          <InheritanceItem from="scPoli" to="CPA" desc="cVAE → 加性解耦扰动预测" />
          <InheritanceItem from="scBERT" to="Geneformer" desc="BERT 架构 → Rank Encoding 改进" />
          <InheritanceItem from="scPEFT" to="多个 scFM" desc="参数高效微调框架，适用于各类基础模型" />
        </div>
      </section>

      {/* ===== ⑤ 交互式学习路线图 ===== */}
      <section className="overview-section">
        <h2 className="overview-section-title">🧭 交互式学习路线图</h2>
        <p className="overview-section-desc">
          按技术路线组织的学习路径。勾选已学习的笔记追踪进度，数据保存在本地浏览器中。
        </p>
        <LearningRoadmap onNavigate={onNavigate} />
      </section>
    </div>
  )
}

/* ===== 子组件 ===== */

function LineageCard({ tree, onNavigate }: { tree: TechNode; onNavigate: (id: string) => void }) {
  return (
    <div className="lineage-card">
      <div className="lineage-card-header">
        <h4>{tree.name}</h4>
        {tree.description && <span className="lineage-desc">{tree.description}</span>}
      </div>
      <div className="lineage-items">
        {tree.children?.map((child) => {
          const noteId = child.noteId
          return noteId ? (
            <button
              key={noteId}
              className="lineage-item"
              onClick={() => onNavigate(noteId)}
            >
              <span className="lineage-item-name">{child.name}</span>
              {child.description && (
                <span className="lineage-item-desc">{child.description}</span>
              )}
            </button>
          ) : (
            <div key={child.name} className="lineage-item lineage-item-placeholder">
              <span className="lineage-item-name">{child.name}</span>
              {child.description && (
                <span className="lineage-item-desc">{child.description}</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function LabCard({ group, onNavigate }: { group: LabGroup; onNavigate: (id: string) => void }) {
  return (
    <div className="lab-card">
      <h4 className="lab-card-name">{group.name}</h4>
      <div className="lab-card-members">
        {group.members.map((id) => {
          const meta = noteMetas.find((n) => n.id === id)
          return meta ? (
            <button key={id} className="lab-member" onClick={() => onNavigate(id)}>
              {meta.title}
            </button>
          ) : null
        })}
      </div>
    </div>
  )
}

function InheritanceItem({ from, to, desc }: { from: string; to: string; desc: string }) {
  return (
    <div className="inheritance-item">
      <span className="inheritance-from">{from}</span>
      <span className="inheritance-arrow">→</span>
      <span className="inheritance-to">{to}</span>
      <span className="inheritance-desc">{desc}</span>
    </div>
  )
}
