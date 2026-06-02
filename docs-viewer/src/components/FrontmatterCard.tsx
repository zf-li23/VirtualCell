import type { FrontMatter } from '../hooks/useNotes'

interface Props {
  frontmatter: FrontMatter
}

function RepoLink({ url }: { url: string }) {
  const host = url.replace(/^https?:\/\//, '').split('/')[0]
  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className="fm-link">
      {host}
    </a>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    done: '\ud83d\udfe2',
    metadata: '\ud83d\udfe1',
    template: '\u26aa',
  }
  return <>{colors[status] ?? '\u26aa'} {status}</>
}

export function FrontmatterCard({ frontmatter }: Props) {
  if (!frontmatter.status && !frontmatter.filled && !frontmatter.code_url) {
    return null
  }

  return (
    <div className="frontmatter-card">
      {frontmatter.status && (
        <div className="fm-row">
          <span className="fm-label">状态</span>
          <span className="fm-value"><StatusBadge status={frontmatter.status} /></span>
        </div>
      )}
      {frontmatter.filled && (
        <div className="fm-row">
          <span className="fm-label">撰写日期</span>
          <span className="fm-value">{frontmatter.filled}</span>
        </div>
      )}
      {frontmatter.category && (
        <div className="fm-row">
          <span className="fm-label">分类</span>
          <span className="fm-value">{frontmatter.category}</span>
        </div>
      )}
      {frontmatter.code_url && (
        <div className="fm-row">
          <span className="fm-label">代码仓库</span>
          <span className="fm-value"><RepoLink url={frontmatter.code_url} /></span>
        </div>
      )}
    </div>
  )
}
